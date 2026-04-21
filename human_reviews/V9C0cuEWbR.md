# Soft Convex Quantization: Revisiting Vector Quantization with Convex Optimization

- Avg Score: 4.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 6, 3, 3

## Abstract
Vector Quantization (VQ) is a well-known technique in deep learning for extracting informative discrete latent representations. VQ-embedded models have shown impressive results in a range of applications including image and speech generation. VQ operates as a parametric K-means algorithm that quantizes inputs using a single codebook vector in the forward pass. While powerful, this technique faces practical challenges including codebook collapse, non-differentiability and lossy compression. To mitigate the aforementioned issues, we propose Soft Convex Quantization (SCQ) as a direct substitute for VQ. SCQ works like a differentiable convex optimization (DCO) layer: in the forward pass, we solve for the optimal convex combination of codebook vectors that quantize the inputs. In the backward pass, we leverage differentiability through the optimality conditions of the forward solution. We then introduce a scalable relaxation of the SCQ optimization and demonstrate its efficacy on the CIFAR-10, GTSRB and LSUN datasets. We train powerful SCQ autoencoder models that significantly outperform matched VQ-based architectures, observing an order of magnitude better image reconstruction and codebook usage with comparable quantization runtime.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes a Soft Convex Quantization (SCQ) scheme to substitute the vector quantization (VQ) module in generative modeling. The authors first outline the challenges in VQ, including gradient approximation, codebook collapse, and lossy quantization, as well as recent related works for alleviating these issues. The proposed SCQ leverages convex optimization to perform soft quantization, which acts as an improved drop-in replacement for VQ that addresses many of the challenges. Experimental results demonstrate that SCQ is superior to existing VQ methods on various datasets.

### Strengths
1. The paper is well written and easy to follow.
2. The proposed SCQ is a novel method to mitigate the challenges in VQ mentioned above. First, the optimization of SCQ (via Eq. (8)) selects a convex combination of codebook vectors for each embedding, which means that the codebook capacity can be sufficiently utilized to avoid codebook collapse. Second, SCQ is implemented via Differentiable Convex Optimization (DCO), which enables effective and efficient backpropagate through soft quantization.
3. Extensive experiments are conducted to validate the effectiveness of the proposed SCQ. SCQ outperforms recent state-of-the-art methods on two different datasets in image reconstruction task in terms of three measurements. Analysis of SCQ presents that SCQ improves codebook perplexity and convergence during training. Besides, when combined with the training first-stage models, the authors show that SCQGAN performs better than VQGAN.

### Weaknesses
1. Soft quantization is also used to solve the gradient approximation challenge discussed in section 2.2.1, which is not mentioned in the paper.
2. The authors claim that SCQ is implemented with DCO. However, the relationship between the optimization of SCQ and DCO is ambiguous, especially the relationship between Eq. (7) and Eq. (8).
3. The analysis of how SCQ addresses the lossy quantization challenge compared with existing VQ methods is not convincing. More theoretical analysis or experiments are encouraged to demonstrate this point.
4. Some details are missing. For example, in Eq.(8), the codebook used to obtain \tilde(P) is not specified. Besides, in section 4.2, the training loss of SCQGAN is not given.

### Questions
1. What is the relationship between the optimization of SCQ and DCO?
2. Could the authors provide more detailed analysis on how SCQ mitigates the lossy quantization challenge?
3. Could the author give the missing details mentioned in weaknesses #4?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a differentiable and convex optimization layer as a direct substitute for vector quantization (VQ). It addresses three major challenges in classical VQ: a) non-differentiable k-means is replaced by softmax and gumbel sampling; b) codebook collapse through stochasticity; c) transforms the NP-hard quantization centroid generation into a convex hull over codebook vectors. The results outperforms other VQ-based architectures for image processing tasks.

### Strengths
1. This work demonstrated a great alternative to k-means based VQ training which is differentiable and convex. 

2. Clean formulation on the optimization goals and algorithms.

3. Clear performance wins over other VQ techniques for neural networks.

### Weaknesses
1. In each comparison table, the non-quantized baseline numbers are missing.

2. Author emphasizes the problem of codebook collapse, but there were no quantitative support for how SCQ performs better than traditional k-means or VQ.

### Questions
1. Please elaborate on how SCQ prevents the codebook collapse problem.

2. What is the runtime latency of this work when compared to the uncompressed baseline?

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose soft convex quantization (SCQ) as a drop-in replacement for vector quantization (VQ) for generative modeling with architectures such as VQVAEs and VQGANs. 

SCQ learns a codebook of vectors similar to VQ. However, SCQ uses its code vectors to define a convex polytope. Then, to "quantize" an arbitrary vector, SCQ projects it onto the closest point of this convex polytope. In particular, SCQ "quantizes" vectors inside the polytope with essentially zero error. This approach starkly contrasts with VQ, which quantizes an arbitrary vector to its closest code vector.

While SCQ's projection operation is differentiable (as opposed to VQ, which needs to use a straight-through estimator to perform gradient descent-based optimization), the authors note that it is not scalable. Thus, they develop a scalable approximation to it.

The authors perform image reconstruction experiments on several datasets and show that their method compares favorably to VQ-based methods.

### Strengths
I found using the code vectors to define a convex polytope and "quantize" vectors by projecting them onto the polytope quite interesting.

Furthermore, the authors compare their methods across several datasets using several different generative models.

### Weaknesses
I also find soft convex quantization a misnomer because the method doesn't quantize vectors (i.e., it doesn't map them to a discrete set of representations); it projects them onto a polytope. This latter issue is especially problematic because it means that comparing SCQ to VQ is meaningless.

From the perspective of autoencoders, SCQ is a worse model because it has zero error inside its convex polytope, but it introduces a projection error outside of it. Furthermore, it is computationally more expensive to use SCQ than just using a standard autoencoder.

However, theoretical issues aside, my primary problem with the paper is that the model's use cases are unclear. 

As I understand, VQ is usually used for generative modeling or data compression. Since SCQ produces continuous "quantized" representations, it cannot be used for compression, so its use case seems to be restricted to generative modeling. However, the authors do not perform any experiments on generative modeling, only image reconstruction, so SCQ's performance for this task is unclear.

Could the authors please correct me if I misunderstood their approach? If not, could they please clarify what use case they intend for their method?

Besides this, the writing needs to be improved. The authors use non-standard terminology, e.g. they use the terms "first-stage" and "second-stage" models for inference and generative networks, respectively. There are many strangely worded sentences that are difficult to understand.

They also seem to misuse the term "sparsity" in section 3.1. A sparse vector has most of its entries be exactly zero, while the authors seem to mean close to zero.

In the same paragraph, the symbol C is overloaded to mean the number of input channels and the codebook matrix.

Finally, the authors don't provide any formal analysis for or perform any ablation studies between the "proper" convex optimization procedure in Algorithm 1 and the relaxed version they propose in Algorithm 2, so the error introduced by this approximation is unclear.

### Questions
n/a

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This work proposes a continuous relaxation of vector quantization, where rather than assigning a vector to a single codebook element, the vector can be assigned to anywhere within the convex hull of codebook elements. This makes the resulting VQ variant easier to integrate into SGD-based training routines, increases codebook utilization due to the use of soft assignments, and increases the quality of the quantized representation.

### Strengths
1. Strong experimental results and useful illustrations for image reconstruction
1. Strong mathematical exposition

### Weaknesses
1. The comparison against VQ-based quantizers for image reconstruction quality needs to be justified more. Using VQ on a K-vector codebook leads to $\log K$ bits per codeword, while this technique seems to use $O(K)$ bits, due to its soft assignments (unless the assignments are rounded later on, which I don't believe is the case). It therefore seems rather obvious and trivial that this much higher-bitrate representation leads to better image reconstruction; it simply isn't much of an information bottleneck, relative to traditional VQ. Either:

    * Justify the comparison on image reconstruction quality, when the baseline algorithms seem to be at a severe disadvantage given their much lower bitrate.
    * Show superior results on downstream tasks such as image generation, where compressed representation isn't the goal but rather a means to achieve the desired result.
1. No mention of training time. The scalable relaxation's runtime of $O(K^3)$ still seems rather expensive. How does this compare to traditional VQ-based methods?
1. Minor weakness in notation: $C$ is both the codebook and the number of channels.

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

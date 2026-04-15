# AffineQuant: Affine Transformation Quantization for Large Language Models

- Decision: Accept (poster)
- Scores: 8, 5, 8, 8

## Abstract
The significant resource requirements associated with Large-scale Language Models (LLMs) have generated considerable interest in the development of techniques aimed at compressing and accelerating neural networks. 
Among these techniques, Post-Training Quantization (PTQ) has emerged as a subject of considerable interest due to its noteworthy compression efficiency and cost-effectiveness in the context of training. 
Existing PTQ methods for LLMs limit the optimization scope to scaling transformations between pre- and post-quantization weights. 
This constraint results in significant errors after quantization, particularly in low-bit configurations. 
In this paper, we advocate for the direct optimization using equivalent Affine transformations in PTQ (AffineQuant). 
This approach extends the optimization scope and thus significantly minimizing quantization errors. 
Additionally, by employing the corresponding inverse matrix, we can ensure equivalence between the pre- and post-quantization outputs of PTQ, thereby maintaining its efficiency and generalization capabilities. 
To ensure the invertibility of the transformation during optimization, we further introduce a gradual mask optimization method. 
This method initially focuses on optimizing the diagonal elements and gradually extends to the other elements. 
Such an approach aligns with the Levy-Desplanques theorem, theoretically ensuring invertibility of the transformation. 
As a result, significant performance improvements are evident across different LLMs on diverse datasets. 
Notably, these improvements are most pronounced when using very low-bit quantization, enabling the deployment of large models on edge devices. 
To illustrate, we attain a C4 perplexity of $15.76$ (2.26$\downarrow$ vs $18.02$ in OmniQuant) on the LLaMA2-$7$B model of W$4$A$4$ quantization without overhead. 
On zero-shot tasks, AffineQuant achieves an average of $58.61\%$ accuracy ( $1.98\%\uparrow$ vs $56.63$ in OmniQuant) when using $4$/$4$-bit quantization for LLaMA-$30$B, which setting a new state-of-the-art benchmark for PTQ in LLMs. 
Codes are available at: https://github.com/bytedance/AffineQuant.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This work develops a quantization method that uses an affine matrix transformation to quantize the weights of a transformer. By doing so, the objective function has more parameters to be optimized and can achieve lower quantization error when the weight matrix is compressed to a low-bit representation (2-bit). 

Recommendation: The presentation and results are currently misleading. I might miss something, but currently, the method has no practical benefit. If the authors can explain to me what I am missing, I am happy to raise my score significantly (2-4 points). I will also raise my score by 2 points if the presentation of 2-bit results is dropped in favor of an analysis of information density (scaling curves that show Pareto fronts in terms of performance per bit of total memory footprint).

### Strengths
The results from affine quantization are strong compared to baselines.

### Weaknesses
- The paper misleadingly highlights 2-bit quantization when the same paper shows higher information density for 4-bit quantization (4-bit + 7B params > 2-bit + 13B params, etc.). This is a mistake that is propagated in the literature. I will not accept work that presents quantization results like this because it is very misleading.
- It is unclear what the benefit of the method is. To use the quantization in practice, the inverse affine transformation matrix is needed to convert the input tensor. However, by having to store this matrix, the size of the model is doubled, which negates FLOPs and memory benefits. Am I missing something?

### Questions
- How does the process work at inference time? If you optimize with the affine matrix during quantization, you surely need it during inference time.
- How fast is inference for this method compared to baselines?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces "AffineQuant", a method to optimize Large-scale Language Models (LLMs) using equivalent Affine transformations in Post-Training Quantization (PTQ). Traditional PTQ techniques often resulted in significant errors, especially in low-bit configurations. AffineQuant expands the optimization scope, reducing these errors. Using a unique gradual mask optimization method aligned with the Levy-Desplanques theorem, invertibility of transformations is ensured. The results show AffineQuant outperforms existing methods, especially in low-bit configurations, making it a promising tool for model compression and deployment.

### Strengths
- This paper is well-written and successfully exhibit their features.
- This paper extends current PTQ papers to introduce affine transformation to MRE (minimum reconstruction error) methods. It could be reasonable in the line of quantization papers' history.

### Weaknesses
- The paper appears to overlook contemporary quantization methodologies like FlexRound. Notably, FlexRound seems to be a successor in the lineage of MRE-PTQ techniques, such as BrecQ and QDrop, designed for LLM compression. A comparative analysis with these methods would accentuate the novelty and efficacy of the proposed technique. Reference for consideration: https://arxiv.org/abs/2306.00317.

- Regarding Figure 3, my understanding is that the loss of a partial layer during MRE PTQ might not directly correlate with improved model performance, especially in the context of generative AI models.

- I have a few observations concerning Section 4, which details experimental results:
   1) A significant concern is the sole reliance on PPL scores as a performance metric. While it corresponds to dataset loss, recent LLM compression research suggests that PPL may not fully capture the generative capabilities post-quantization. It might be worthwhile to consider metrics like common-sense reasoning or evaluations using the MMLU dataset.
   2) Additionally, an increase in PPL scores by over 10 points indicates a significant degradation in the generation capabilities of the quantized model. Thus, contrasting the performance of such models, like the W2A16 results mentioned in the abstract, might not provide meaningful insights. A closer look at the actual generation outputs from these severely quantized models could reveal issues like non-coherent results or repeated verbiage.
Lastly, larger models could yield different outcomes, as they might exhibit distinct patterns of outliers or weight distributions.

### Questions
included in weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Propose affine transformation in PTQ (in line with "equivalent transformations" line of work) based on specialized optimization approach involving a “gradual masking” to ensure a viable affine matrix is trained.

### Strengths
- A more general approach than several proceedings works in the space of “equivalent transformations”, and a nice explanation in Section 3.1
- strong empirical results

### Weaknesses
- I imagine there is an increased computational cost to the proposed method. Is that the reason why results on larger OPT (ie 66B) or Llama (70B) models were not reported? My understanding is that this is the main tradeoff: a more powerful quantization method, but at an increased computational cost?

### Questions
- In terms of a computation vs quantization performance tradeoff, what do the authors think about just fine-tuning the model after some standard quantization method? I understand how the gradual mask training approach makes this cheaper than fine-tuning the full model, but it still appears expensive enough that it’s difficult to get results on the largest OPT/Llama models. I think this is an interesting method, more of a point of discussion.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces a method for post-training quantization of Large Language Models (LLMs), specifically focusing on the linear layers. This quantization is uniform, employing fixed bit widths across the layers. The proposed approach involves learning an affine transformation (which is more general than single scaling) of weights before quantization by minimizing the MSE loss. The authors aim to ensure that this matrix remains non-singular, employing the Levy-Desplanques theorem and their proposed Gradual Mask (GM) method. Experiments are conducted on various model variants, including OPT and LLaMA, where the proposed algorithm is comparable or outperforms previous works, especially in low bit-width configurations.

=================================

Update: 
The authors addressed my concerns, and I have increased the given rating correspondingly.

### Strengths
1. The paper expands on pre-quantization transformations by introducing invertible affine transformations, which are more versatile than previous methods that rely on simple scaling.
2. The proposed post-training quantization method for LLMs addresses a timely and pressing need due to the growing popularity of LLMs with their large model sizes.
3. The method's ability to avoid model retraining is a substantial practical advantage since retraining LLMs can be computationally expensive and time-consuming.
4. The weight transformation before quantization is not limited to LLMs; it can be applied to linear layers in different models as well.
5. Inference speed is unaffected, as the additional matrices can be fused with the weight matrices.
6. The paper provides extensive experimental results, including comparisons across different datasets and NLP models.

### Weaknesses
1. A significant weakness of this paper is the lack of clarity in explaining the implementation of the core concept, which involves the use of strictly diagonal matrices and the proposed Gradual Mask (GM). Figure 2 suggests that the GM matrix is element-wise multiplied by the matrix A, but the description implies a different interpretation, where it functions as a learning rate for each element in A. This discrepancy needs further clarification to provide a complete understanding of the method.

2. The hyper-parameters $b$ (bit-width) and $\alpha$ (stability factor) may introduce significant computational overhead in the pursuit of determining the optimal trade-off between model size and accuracy.

### Questions
1. The method employs non-differentiable components in its optimization approach. It would be interesting to understand how the authors address this challenge in practice.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

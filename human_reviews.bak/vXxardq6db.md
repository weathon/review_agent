# SliceGPT: Compress Large Language Models by Deleting Rows and Columns

- Decision: Accept (poster)
- Scores: 6, 6, 6, 5, 5

## Abstract
Large language models have become the cornerstone of natural language processing, but their use comes with substantial costs in terms of compute and memory resources. Sparsification provides a solution to alleviate these resource constraints, and recent works have shown that trained models can be sparsified post-hoc. Existing sparsification techniques face challenges as they need additional data structures and offer constrained speedup with current hardware. In this paper we present SliceGPT, a new post-training sparsification scheme which replaces each weight matrix with a smaller (dense) matrix, reducing the embedding dimension of the network. Through extensive experimentation we show that SliceGPT can remove up to 25% of the model parameters (including embeddings) for LLAMA-2 70B, OPT 66B and Phi-2 models while maintaining 99%, 99% and 90% zero-shot task performance of the dense model respectively. Our sliced models run on fewer GPUs and run faster without any additional code optimization: on 24GB consumer GPUs we reduce the total compute for inference on LLAMA-2 70B to 64% of that of the dense model; on 40GB A100 GPUs we reduce it to 66%. We offer a new insight, computational invariance in transformer networks, which enables SliceGPT and we hope it will inspire and enable future avenues to reduce memory and computation demands for pre-trained models.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors of this paper describe their methodology as a transformation of a Transformer network from LayerNorm to RMSNorm. They implement an approach involving the application of orthogonal-matrix transformations and the selective removal of columns and rows from the transformed weight matrices. This process is aimed at reducing the overall model size while preserving performance integrity. The results of their research demonstrate a significant improvement in perplexity on benchmark datasets, OPT and Llmas2, aligning with the 2:4 scheme and underscoring the substantial enhancement in model efficiency and accuracy.

### Strengths
* The formulation is clear and enhanced by illuminating diagrams for better comprehension.
* The experimental results illustrate the method's effectiveness, establishing a well-defined trade-off between accuracy and sparsity.

### Weaknesses
The experimental section has certain shortcomings:

1. The experiment section does not comprehensively address the comparison between SliceGPT and SparseGPT. While 2:4 sparsity implies a 50% compression rate, Table 1 exclusively showcases SliceGPT with up to 30% compression. This limitation hinders a clear conclusion regarding the superior performance of SliceGPT over SparseGPT.

2. The absence of inference time data for SparseGPT in the experiments makes it challenging to convincingly demonstrate the superior efficiency of SliceGPT.

3. The paper lacks a comparative analysis with state-of-the-art pruning methods such as low-rank approximation, unstructured sparsity, and block sparsity. The omission of these comparisons limits the paper's ability to establish the competitiveness of SliceGPT within the broader context of pruning techniques.

### Questions
* Can you show the performance (perplexity, inference time) of SliceGPT at 50% sparsity?
* Can you show the inference time of SparseGPT in comparsion to SliceGPT under the same experimental setup?

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
This paper introduces SliceGPT, a method to reduce the size of matrices for inference of LLMs. The method uses orthogonal matrices to project to a lower-dimensional space the weight matrices, these orthogonal matrices being constructed using PCA.

### Strengths
- The paper is well written and pleasant to follow. Ideas are simply explained and figures are helping the understanding. 
- Experimental results are convincing.
- I think this method could be really used in practice to reduce inference time.

### Weaknesses
- I think section "layernorm transformers can be converted to RMSnorm" is not well motivated. Could the authors explain more in details the subtleties of this section and why it was written? I may have missed the point.
- I'll wait for other reviewers weaknesses to see whether I agree with them.

### Questions
- Do the authors plan to release the code? I think open sourcing it is very important for the community.
- The latex is broken, citations are not redirecting, I think your should recompile the pdf.
- Could the authors comment on the use of a random projection (which is orthogonal in expectation, as in sketching methods) compared to $Q_\ell$ computed using by PCA, which is more expensive?
- In practice, not all layers may be equivalent signal-wise, could the authors comment on the possible use of weight watcher ( https://github.com/CalculatedContent/WeightWatcher ) to analyze how to select a different projection dimension for each layer? This question is purely curiosity but I think, combining both SliceGPT and weight watcher could greatly improve the method.
- p8: what do the authors mean by "using dense kernels in our compressed models"? Did they code specific kernels for SliceGPT?
- I think the authors should write a small proof of Equation (2) to increase the readability of the paper. Can the authors provide it in their answer?
- "Theorem" is too strong for Theorem 1, I suggest "Proposition" or "Remark".
- p4: typo: OBC instead of OBS.

Overall I liked the paper and the method, and satisfying answers to my questions and weaknesses would make me consider increase my score.

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a technique for pruning neurons in Tranformer architectures based on a clever application of orthogonal matrices, which enables PCA-based elimination of rows and columns throughout the architecture. The authors evaluate their technique on a range of large language models from the OPT and Llama-2 families and demonstrate improvements in inference runtime on GPUs.

### Strengths
The paper was very well written and organized. I found the method easy to understand. The insights that underpin the method (e.g., invariance to repeated application of orthogonal matrices) are clever and I think the PCA-based pruning of rows and columns in weight matrices is nicely grounded relative to other neuron pruning techniques.

### Weaknesses
I think there are two main weaknesses in this paper. First, the authors don’t acknowledge prior work on neuron pruning. Admittedly most of the papers that I’m aware of on this topic focus on convolutional neuron networks. But, some of the methods are likely to provide a reasonable baseline for the proposed technique. I’ve cited some potentially relevant papers below [1, 2, 3, 4, 5].

Second, the results in Table 1 suggest to me that 2:4 sparsity is preferable to the proposed technique? If I understand correctly, 2:4 will remove 50% of the weights in the model and the results in Table 1 show that it suffers less quality degradation than removing 30% of the parameters with SliceGPT. Based on this, I expect 2:4 sparsity would show larger inference runtime savings for a given quality than the results in Table 2.

[1] https://arxiv.org/abs/1708.06519

[2] https://arxiv.org/abs/1707.06342

[3] https://arxiv.org/abs/1707.01213

[4] https://arxiv.org/abs/1707.06168

[5] https://arxiv.org/abs/1810.05270

### Questions
I have no additional question.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces SliceGPT - a new approach for compressing large language models. By deleting rows and columns based on computational invariance, SliceGPT can significantly reduce the computation and memory required for inference while maintaining high accuracy. The evaluation demonstrates that this method is effective for large models such as OPT-66B and Llama-70B.

### Strengths
- A novel method of compression based on computational invariance
- No special code is required to run the compressed models and achieve speedup and memory savings
- Works for Llama-70B and OPT-66B
- It is well-written and easy to follow

### Weaknesses
- The accuracy loss is not "negligible". With 25% sparsity, the perplexity of Llama-2-70B on WikiText2 increases from 3.32 to 4.89, which is similar to a dense Llama-2-13B. However, a 25% sparse Llama-2-70B has much more parameters than a dense Llama-2-13B.
- The speedup is not impressive.
- Compared to the quantization-based method, there is no advantage.

### Questions
1. In Table 2, it is not fair to multiply the number of GPUs by the total latency and get "GPUms". Huggingface Transformers implements naive model parallelism (or device placement, or pipeline parallelism without pipelining) method to parallelize the models, which means that only one GPU is active at a time. A correct implementation of tensor parallelism or pipeline parallelism will give different results. Considering this, the latency speedup is less impressive.
2. Give the same parameter count budget or inference latency budget, how does this method compare to quantization-based method?
3. The "computational invariance" trick is similar to a trick in SmoothQuant[1] (equation 3). Both of them multiplicate some matrices between the X and W, so it is good to do some comparison here.

[1] SmoothQuant: Accurate and Efficient Post-Training Quantization for Large Language Models.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 5

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
The paper proposes to idea of using computation invariance for row-/column-wise sparsification. The authors leverages the idea of pre- and post-multiplying each block in a transformer model by orthogonal matrices that warrants computational invariance of each block. On the surface, adding new operations increases the raw FLOPs. However, following this technique, the authors show that they can sparsify most of operations in a transformer models, including attention and FFN layers.

### Strengths
$\mathtt{+}$ The idea of computational invariance and re-purposing additional computation for higher opportunity for sparsification is interesting and warrants further investigation.

$\mathtt{+}$ The results are promising and show the benefits across a range of SOTA models. The comparison with SparseGPT technique is also valuable.

### Weaknesses
$\mathtt{-}$ The paper lacks sufficient insights of how the rows and columns are sparsified. It was not clear whether some operations are friendlier to row vs. column sparsification or this is a byproduct of the computational invariance approach.

$\mathtt{-}$ The paper compares accuracy with 2:4 structured sparsity but does not provide head-to-head comparison with SparseGPT (2:4) in terms of latency. 

$\mathtt{-}$ One of the premises of the paper is memory saving, but going through the results it is not clear how the memory savings are in comparison to 2:4 sparsity. Showing a trade-off possibly can clarify this point.

### Questions
I think if the authors could clarify the following questions/comments and include few additional results, the quality of the paper could significantly increase:

(Q1) Show latency comparison across different baselines, (a) Dense, (b) SliceGPT, (c) SparseGPT. 

(Q2) I may have missed this in the paper, but can you please clarify how you decide on row/column sparsity and how you select them? If the sparsed rows/columns are spread across the matrix, how do you manage to do the multiplication while getting latency benefits? or the overall benefits are derived from memory savings?

(Q3) Do you have any insights as which operation/layer is more sensitive to sparsification? Have you thought of not uniformly sparsifying all the layers? Can looking into the range of values in the weight matrices provide insights on how to apply the sparsificiation (both degree and pattern)?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

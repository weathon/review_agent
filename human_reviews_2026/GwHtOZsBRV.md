# LAVA: A UNIFIED FRAMEWORK FOR FINETUNING LANGUAGE AND VISION MODELS

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
LoRA and its variants have attracted considerable attention because of their abilities to tune a negligible number of parameters while achieving comparable downstream performance. This success is largely attributed to the intrinsic low-rank structure of model parameter spaces, which allows LoRA to train two projection matrices to project weights into a low-dimensional subspace and then map them
back. However, it does not consider how to explore this low-rank subspace sufficiently and may lose the expression ability accordingly. Moreover, when using LoRA to tune convolution layers, a flatten operation is required to convert tensors into matrices. We argue that this will degrade the model’s performance. In this paper, we address this issue from a general parameter sub-space perspective: we present a unified \textbf{L}anguage \textbf{A}nd \textbf{V}ision \textbf{A}daption finetuning framework (called LAVA). Specifically, we verify the existence of low-rank subspaces in convolution layers empirically and propose to parameterize the increment of both convolution kernels and matrices as a sum of learnable rank-1 components. To improve training stability, we analyze the optimization dynamics of LoRA and incorporate orthogonal regularization into our parameterization, for which we give theoretical proof that it will help reduce the variance of the gradient. We conduct various experiments on different downstreaming tasks to validate LAVA’s superiority. For example, when tuning LLaMA2-7b for commonsense tasks, the performance of our LAVA is $\textbf{+1.9}$% higher than that of LoRA. For metric depth estimation tasks, LAVA only tunes $\sim$1.5\% of Depth-Anything\textsubscript{large} (335.3M), and achieves $\textbf{+3.5}$% $\delta_1$ accuracy against that of LoRA and $\textbf{+5.6}$% $\delta_1$ accuracy against that of SVDiff.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper proposes a parameter-efficient finetuning (PEFT) method named LAVA, which applies Tucker and CANDECOMP/PARAFAC decompositions to reparameterize the weights in convolutional layers. This design preserves the original weight shapes and structural properties of convolutional filters. To enhance training stability, the authors further introduce an orthogonal regularization term in the parameterization. Experiments are conducted on a variety of NLP and vision benchmarks, including language understanding, commonsense reasoning, and image segmentation, showing (limited) improvements over several baselines.

### Strengths
The main idea is interesting: decomposing convolutional weights while maintaining their structural form is non-trivial and practically meaningful.

Incorporating orthogonal regularization into the decomposition framework is a reasonable design choice that could potentially improve training stability and generalization.

### Weaknesses
1. In Equation (1), the variables *x* and *y* are not explicitly defined, which makes the formulation hard to follow.
2. The decomposition described in Section 3 is somewhat unconventional. Typically, for convolutional weights $W'$, the decomposition is applied over dimensions $c_{out}$ and $c_{in} \times h \times w$, rather than $c_{out} \times h$ and $c_{in} \times w$. The motivation and benefit of this alternative factorization are unclear, and this design choice makes the comparisons in Section 3 less convincing.
3. The experiments use **LLaMA-2-7B** as the main backbone. Given the availability of more recent models (e.g., LLaMA-3, Mistral, or Gemma), this weakens the empirical support for the claimed generality and advancement of LAVA.
4. The paper claims that orthogonal regularization improves training stability, but there are no explicit experiments or analyses (e.g., training curves or variance metrics) to support this claim.
5. The paper does not compare against several relevant and recent PEFT methods, such as **FLoRA**, **DoRA**, and **Conv-LoRA**. Tables 2 and 3 omit many state-of-the-art baselines, making it difficult to assess the true contribution and competitiveness of LAVA.

### Questions
See weaknesses.

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
LAVA proposes a unified, parameter-efficient fine-tuning framework aimed at addressing two core issues in existing LoRA-based methods for vision and language tasks:

Insufficient exploration of low-rank subspace: the optimization process of LoRA may lead to redundant dimensions

Improper handling of convolution layers: flattening operations destroy spatial encoding properties

Key contributions include:

Tensor-factorization perspective: parameterizes convolutional kernel updates as a sum of rank-1 tensor components, preserving full dimensional integrity

Orthogonal regularization: theoretically shown to reduce gradient variance and stabilize training

Unified framework: applicable to both attention mechanisms and convolutional networks

### Strengths
Conducts the first systematic analysis of low-rank subspace properties in convolution layers within the PEFT paradigm

Provides rigorous theoretical proofs for orthogonal regularization (Theorem 1 & Proposition 1)

Establishes the mathematical connection that shows LoRA is a special case of LAVA

Covering NLU, commonsense reasoning, semantic segmentation, depth estimation, and text generation

### Weaknesses
The evaluation does not include several recent and competitive PEFT baselines, such as LoRA+, VeRA, DoRA, and NoRA.

The computational efficiency analysis is incomplete, as it lacks direct comparisons of training time and memory usage.

Suggestion: include comparisons with more state-of-the-art methods in the revised version.

### Questions
same as above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces LAVA (Language And Vision Adaption), a unified framework for the parameter-efficient fine-tuning of large models. The authors identify two primary limitations with the widely-used LoRA method: (1) Subspace Redundancy, where unconstrained training can lead to correlated, inefficient representations in the low-rank update, and (2) Dimension Disorder, where applying LoRA to convolutional layers requires flattening tensors into matrices, thereby disrupting the inherent spatial structure of the weights.

LAVA addresses these issues by introducing (1) a generalized subspace-based adaptation that handles high-order tensors (like convolution kernels) directly by parameterizing the weight update as a sum of rank-1 tensors, thus preserving dimensional integrity. This method naturally reduces to the LoRA formulation when applied to matrices. (2) A column-orthogonal regularization term applied to the trainable low-rank matrices. This encourages the basis vectors to be orthogonal, promoting a more complete exploration of the low-rank subspace and, as the authors show theoretically, stabilizing training by reducing the variance of gradients.

The authors conduct a comprehensive set of experiments across natural language understanding (GLUE), commonsense reasoning (LLaMA2-7B), semantic segmentation (SAM), depth estimation (Depth-Anything), and text-to-image generation (SDXL). Their results consistently show that LAVA outperforms LoRA and other PEFT baselines.

### Strengths
The paper is built on a very clear critique of LoRA. The concepts of dimension disorder for convolutions and dimension redundancy from unconstrained optimization are well-explained, and represent meaningful limitations in current PEFT approaches. 

LAVA is a simple but effective framework that simple addresses the identified problems. Using a tensor decomposition-inspired update for convolutions is a natural fit, and extending LoRA with orthogonal regularization is a principled way to improve subspace exploration. 

The empirical evaluation compares LAVA and LoRA across many tasks, further supporting the efficacy of LAVA.

### Weaknesses
The core components of LAVA (tensor decomposition [3,4] and orthogonal regularization) are not novel in isolation. Orthogonal constraints are a well-known tool in machine learning for improving training stability and representation quality and have been previously used for PEFT [1, 2]. The paper would be strengthened by a more detailed discussion of related work that has used similar techniques, even outside the direct context of PEFT, to better contextualize its specific contribution. For instance, the distinction from OFT could be sharpened.

The work lacks comparison (or integration ?) into more recent PEFT alternatives to LoRA [5,6,7] and others. Do these state-of-the-art PEFT approaches also suffer from subspace redundancy and dimension disorder also or are some of these problems already partly addressed. In this case, is LaVA complementary with these existing solutions ?

The text-to-image generation experiment (Sec 5.5) feels less thorough than the other experimental sections. The evaluation relies primarily on a single qualitative figure in the main paper and one FID score in the appendix. Given the stochastic nature of generation, strengthening this section with more quantitative metrics (e.g., CLIP scores), a user study, or at least more generated examples in the appendix would make the claims in this domain more robust.

[1] Xiao Wang, Tianze Chen, Qiming Ge, Han Xia, Rong Bao, Rui Zheng, Qi Zhang, Tao Gui, and Xuanjing Huang. 2023. Orthogonal Subspace Learning for Language Model Continual Learning. In EMNLP 2023.

[2] Büyükakyüz, K. OLoRA: orthonormal low-rank adaptation of large language models. arXiv preprint arXiv:2406.01775, 2024

[3] Lebedev, Vadim, et al. "Speeding-up convolutional neural networks using fine-tuned cp-decomposition." arXiv preprint arXiv:1412.6553 (2014).

[4] Yifan Yang, Jiajun Zhou, Ngai Wong, and Zheng Zhang. 2024. LoRETTA: Low-Rank Economic Tensor-Train Adaptation for Ultra-Low-Parameter Fine-Tuning of Large Language Models. In Proceedings of Association for Computational Linguistics: Human Language Technologies, 2024

[5] Edalati, Ali, et al. "KronA: Parameter-Efficient Tuning with Kronecker Adapter." Enhancing LLM Performance: Efficacy, Fine-Tuning, and Inference Techniques. Cham: Springer Nature Switzerland, 2025. 49-65.

[6] Liu, Shih-Yang, et al. "Dora: Weight-decomposed low-rank adaptation."  International Conference on Machine Learning. 2024.

[7] Albert, Paul, et al. "RandLoRA: Full-rank parameter-efficient fine-tuning of large models." ICLR (2025).

### Questions
The orthogonal regularization is applied to only one of the factor matrices (U in Eq. 3) in the convolutional case. What was the rationale for this specific choice? Have the authors experimented with regularizing all factor matrices or a different combination, and how did that affect performance and training stability?

Regarding the commonsense reasoning results (Table 7), do the authors have any hypotheses for why LAVA might underperform LoRA on specific datasets like SIQA and WinoGrande? Is it possible that for some tasks, the unconstrained subspace exploration of LoRA is accidentally beneficial, or is it more likely noise?

Could the authors quantify the computational overhead of the orthogonal regularization term? Does it introduce a noticeable slowdown in training wall-clock time compared to a standard LoRA implementation?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The work proposes a method (LAVA) for parameter efficient fine-tuning of vision and language models, particularly focused on convolution layers. Given a tensor, it models the low-rank update as the sum of 'r' CANDECOMP/PARAFAC (CP) rank-1 updates. Each rank-1 update is obtained as an outer-product of learnable vectors. When the tensor is a matrix, this is equivalent to LoRA. Unlike typical application of LoRA to convolution layers by reshaping the weight tensor to be a matrix, the proposed CP rank-1 update in LAVA does not require any reshaping and thus better models spatial information. Additionally, the authors propose orthogonal regularization to ensure the update has maximum possible rank and to improve training stability. The proposed method is shown to outperform baseline approaches including LoRA on diverse vision and language tasks.

### Strengths
1. The idea of using the sum of CP rank-1 updates for convolution layers is interesting. It addresses the specific issue of weight reshaping in LoRA adaptation for convolution.
2. The idea of orthogonal regularization is well motivated. The authors provide theoretical proof for training stabilization due to the proposed regularization.
3. The experiments include diverse tasks on both vision and language models and the proposed method consistently outperforms both LoRA and other baseline approaches.

### Weaknesses
1. The primary contribution of the paper is an effective PEFT method (LAVA) for convolutional layers. However, there is not much discussion or empirical comparison with related PEFT methods focused on convolution like Lora-C [a], Conv-Adapter [b] and LoRAE [c]. Both Conv-Adapter and LoRAE preserve spatial properties in convolution similar to LAVA. Conv-Adapter can be seen as a generalization of LAVA and reduces to LAVA when the learnable 2-D filter is modified to be a separable filter and removing the non-linearity between the depth-wise and point-wise convolution blocks. The experiments are limited with just results with a single dataset and model on depth estimation and image generation tasks. There is no comparison with SOTA approaches on the image generation task. The results on semantic segmentation (table 12 in A.7.2) are not consistent with the existing literature (LoRA consistently outperforms Conv-Lora while Conv-Lora (Zhong et al., 2024) show the opposite on the same datasets).
2. For language models, the proposed approach reduces to applying orthogonal regularization atop LoRA. More experimental results and analysis is required to understand whether and why this is helpful. For instance, does the regularization lead to a higher rank than that observed in LoRA? Or, is the stabler training the reason for performance improvements? The provided LLM results are on just two datasets with just one model (small RoBERTa model on one, LLaMA-2 on another) for each. While I understand the resource requirements for larger scale experiments, more experiments are required for a stronger comparison between LoRA and the proposed method. Discussion and comparison with a related work OLoRA [d] is missing. OLoRA performs orthonormal decomposition of the weight matrix before performing LoRA updates. 

References:

[a] Ding, Chuntao, et al. "LoRA-C: Parameter-Efficient Fine-Tuning of Robust CNN for IoT Devices." arXiv preprint arXiv:2410.16954 (2024). \
[b] Chen, Hao, et al. "Conv-adapter: Exploring parameter efficient transfer learning for convnets." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024. \
[c] Wang, Zhixue, Hongyao Ma, and Jiahui Zhai. "Low-rank adaptation for edge AI." Scientific Reports 15.1 (2025): 33109. \
[d] Büyükakyüz, Kerim. "Olora: Orthonormal low-rank adaptation of large language models." arXiv preprint arXiv:2406.01775 (2024).

### Questions
1. Provide analysis on the rank of learned weight matrices for the language model experiments for both LoRA and LAVA. 
2. The learning rate plays a very important role in LoRA fine-tuning. Since the learning rate is tuned for a particular model and dataset in the 3. LLM experiments, the comparison with LoRA might be unfair. Provide results on language modeling with lr tuning for LoRA. Also, provide details on the dataset split used to perform the lr tuning for LAVA.
4. Provide empirical comparison with Conv-adapter either on existing tasks in LAVA or on the tasks in [b]. 
5. Provide discussion on training compute and memory for LAVA. For LLMs, does splitting the update into `r` matrices significantly increase the training compute and memory compared to LoRA? How does this scale to larger models and ranks?
Add results for multiplier=0 for the plots in Figure 6 (analysis of \lambda). The value of \lambda does not seem to significantly affect results on the NLU tasks and a multiplier of value lower than 1 seems to have the best performance. Why would LAVA then perform significantly better than LoRA?
6. Why are the results for LoRA (encoder+decoder) so much worse than LoRA (encoder) in depth estimation (table 3)? Is it because of the convolutional decoder? Why do we not observe similar degradation in segmentation and image generation tasks? Does this support the use of non-reshaping technique in LAVA? More such experiments on convolution heavy backbones like ResNet would have made the work stronger (not asking for those expts here).

### Soundness
2

### Presentation
3

### Contribution
2

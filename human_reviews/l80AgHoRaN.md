# LoRA-XS: Low-Rank Adaptation with Extremely Small Number of Parameters

- Avg Score: 4.33
- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 3, 5, 5, 5, 3

## Abstract
The rapid expansion of large language models (LLMs) has underscored the need for parameter-efficient fine-tuning methods, with LoRA (Low-Rank Adaptation) emerging as a popular solution. Although LoRA reduces the number of trainable parameters, serving multiple (task or user-specific) LoRA modules on top of a base model still creates significant storage challenges. To address this, using theoretical derivation, we introduce LoRA-XS (Low-Rank Adaptation with eXtremely Small number of parameters), a novel low-rank adaptation method that considerably reduces the trainable parameters while showing superior or competitive performance. LoRA-XS achieves this by inserting a small, trainable $r \times r$ weight matrix between frozen low-rank matrices, which are constructed by Singular Value Decomposition (SVD) of the original weight matrix. This lightweight matrix enables fine-tuning with drastically reduced storage requirements, making it feasible to deploy millions of personalized models while minimizing memory overhead. For instance, LoRA-XS achieves a remarkable reduction of trainable parameters by over 100x in 7B models compared to LoRA. Our evaluations across various benchmarks (including GLUE, GSM8K, MATH, and eight commonsense reasoning datasets) demonstrate that LoRA-XS performs competitively or better than LoRA and other recent methods like VeRA while being significantly more parameter efficient. We also provide an extensive ablation study on the importance of singular vectors in transformer weights, shedding light on the underlying mechanisms driving LoRA-XS’s enhanced efficiency. These findings suggest that LoRA-XS is not only a storage-efficient alternative, but also a powerful tool for scaling and personalizing LLMs at unprecedented scales.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This work introduces LoRA-XS, a novel method for low-rank adaptation that significantly reduces the number of trainable parameters in large language models (LLMs) while maintaining or even improving performance.

### Strengths
1.   LoRA-XS inserts a small, trainable $r \times r$ weight matrix between frozen low-rank matrices constructed from the SVD of the original weight matrix, providing a lightweight mechanism for fine-tuning that requires much less storage space.
2. Despite the significant reduction in parameters, LoRA-XS shows superior or competitive performance across a variety of benchmarks, including GLUE, GSM8K, MATH, and eight common sense reasoning datasets, when compared to LoRA and other recent methods like VeRA.
3. The experiments reveal that self-attention layers can tolerate a high degree of dimensionality reduction, whereas output dense layers benefit from retaining a larger portion of the singular spectrum. This suggests that LoRA-XS offers flexibility in terms of how many and which singular vectors to retain, depending on the specific requirements of the task and the model architecture.
4. The ablation studies provide valuable insights into the role of singular vectors within transformer weights, showing that top singular vectors retain the most task-relevant knowledge, while middle and bottom vectors contribute less to task performance. This finding supports the design choice of using top singular vectors in LoRA-XS.

### Weaknesses
Q1. The performance of LoRA-XS is highly dependent on which singular vectors (top, middle, or bottom) are retained for each module. For example, the self-attention layers may perform well even when only a small fraction of the top singular values is kept, but the output dense layers might require a larger portion of the singular spectrum to maintain good performance. This sensitivity suggests that careful tuning is necessary for optimal results.

Q2. LoRA-XS relies on using SVD to construct the low-rank matrices and then inserting a trainable  $r \times r$ matrix. How to initialize this small matrix and how to choose the rank $r$?

Q3. What is the time overhead of the initial computation of the SVD for the weight matrices of the base model, particularly for very large models?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The pape presents a novel method, LoRA-XS, for parameter-efficient fine-tuning of LLMs. This method is a successor to LoRA, which is popular for reducing the number of trainable parameters during fine-tuning.

The key contributions of LoRA-XS are:

- Extreme Parameter Efficiency: LoRA-XS introduces a small, trainable r × r matrix between frozen low-rank matrices constructed using Singular Value Decomposition of the original weight matrix. This setup significantly reduces the number of trainable parameters compared to LoRA, allowing over 100x reduction in models like GPT-3, making it possible to serve millions of personalized models with minimal memory requirements.

- Performance: The evaluations across multiple benchmarks, such as GLUE, GSM8K, MATH, and eight commonsense reasoning datasets, show that LoRA-XS performs competitively or better than LoRA and recent methods like VeRA, while being significantly more parameter-efficient.

- Independence from Model Dimensions: Unlike existing methods, the number of trainable parameters in LoRA-XS is independent of model dimensions, offering greater flexibility and storage efficiency, especially for large-scale models.

- Ablation Study: The paper also conducts an ablation study on the importance of singular vectors in transformer weights, providing insights into the mechanism that drives LoRA-XS's enhanced efficiency.

Overall, LoRA-XS provides a powerful, storage-efficient approach for scaling and personalizing large language models, maintaining competitive performance with drastically fewer trainable parameters.

### Strengths
**Originality**: The originality of the paper lies in its novel approach to parameter-efficient fine-tuning for LLMs. LoRA-XS builds upon the foundation of LoRA by incorporating a small, trainable matrix inserted between frozen low-rank matrices derived from Singular Value Decomposition. This innovative strategy introduces an extreme reduction in trainable parameters compared to prior methods like LoRA and VeRA. By making the number of parameters independent of model dimensions, the authors have effectively eliminated a significant limitation of existing parameter-efficient fine-tuning techniques. 

**Quality**:The paper is supported by rigorous theoretical derivations and comprehensive empirical evaluations. The authors provide a well-founded theoretical explanation of why LoRA-XS achieves its extreme parameter efficiency by deriving optimal parameter subspaces using truncated SVD. The empirical results on diverse benchmarks, including GLUE, GSM8K, MATH, and commonsense reasoning datasets, demonstrate that LoRA-XS achieves competitive or superior performance to both LoRA and VeRA, despite using significantly fewer parameters.

**Clarity**:The paper is generally well-written and structured, making the proposed method accessible to a broad audience. The use of visual aids, such as diagrams comparing LoRA and LoRA-XS, helps in understanding the differences between these methods.

**Significance**:LoRA-XS attempts to address the significant storage and memory challenges associated with serving millions of personalized models by reducing trainable parameters by over 100x compared to LoRA.

### Weaknesses
There are some weaknesses that impact the overall significance of the work:

1. **Limited Applicability of Memory Savings**: The authors argue that the 100x memory saving benefits the inference of millions of adapters. However, even the most popular model, such as LLaMA3-8B, only has 494 adapters available on Hugging Face ([source](https://huggingface.co/models?other=base_model:adapter:meta-llama/Meta-Llama-3-8B)). This limited usage weakens the practical significance of the proposed memory savings, as the demand for such an extensive number of adapters is currently not evident.

2. **Computation Overhead**: LoRA-XS introduces an additional matrix multiplication per LoRA module, which results in computational overhead during both training and inference. This represents a tradeoff between memory efficiency and computational cost. Unfortunately, the paper does not provide a detailed system footprint analysis to demonstrate the practical tradeoff between memory savings and computational overhead, which is crucial for understanding the overall benefit of the approach.

3. **Inconsistent Performance Across Tasks**: The experimental results show that LoRA-XS does not consistently outperform or match the base models across different downstream tasks. For example, the LLaMA2-7B base model achieves a performance of 77.2 on HellaSwag according to the LLaMA2 paper ([source](https://arxiv.org/pdf/2307.09288)), whereas LoRA-XS achieves a lower score of 75.4 after fine-tuning, indicating negative optimization. Similarly, the performance on BoolQ for the LLaMA2-7B base model is 77.4, but drops to 67.2 after fine-tuning with LoRA-XS. Additionally, ARC-challenge for the LLaMA3-8B base model achieves 78.6, while LoRA-XS results in 76.5, according to the LLaMA3 blog ([source](https://ai.meta.com/blog/meta-llama-3/)). These inconsistencies highlight that LoRA-XS may not provide consistent benefits across all tasks, which reduces its overall reliability and applicability.

4. **Lack of Guidance for Hyperparameter Selection**: The paper lacks advice or experimental evidence on how to choose an appropriate value for the hyperparameter r of the small trainable matrix R. This omission leaves practitioners without clear guidelines for selecting the best configuration for different tasks or models, which could impact the ease of adoption and effectiveness of LoRA-XS.

### Questions
There are some questions for this paper:

1. Gemma 7B base model's performance on MATH is 24.3 according to Hugging Face ([source](https://huggingface.co/google/gemma-7b)). Why is the Full FT performance of 22.74 even lower than the base model?

2. In Table 3, LoRA-XS underperforms compared to LoRA in MATH task. Have you tried using larger ranks for LoRA-XS to improve performance?

3. In Appendix C.2, you train for 2 epochs on MetaMathQA for both LoRA and LoRA-XS, while the typical fine-tuning epoch number is usually 3. How did you choose this hyperparameter, and is it optimal for LoRA or LoRA-XS?

4. In Appendix D, you find that the top subsets of singular vectors in RoBERTa-large retain the most task-relevant knowledge on MRPC, SST2, and MNLI. Does this result hold for different models (e.g., larger models like LLaMA3-70B) and other task scenarios (such as coding and math)? Could you provide some theoretical insights or analysis?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper introduces LoRA-XS, an efficient variant of LoRA aimed at reducing the number of trainable parameters in large language models (LLMs) without compromising the performance. LoRA-XS initializes and freezes the standard low-rank matrices (AB) as top-$r$ singular vectors  derived from SVD of pre-trained weight matrices $W$ and adds trainable $r$ x $r$ matrix between them for flexibility, which is completely independent of the model dimensions. Their approach improves the parameter efficiency by constraining the parameter space for LoRA adaptation from $2nr$ to $r^{2}$, maintaining the model performance in overall. The paper also provides the theoretical derivation that LoRA-XS basically implements an orthogonal projection of any gradient update of $W$ onto a low-rank subspace spanned by the top singular vectors of $W$. The benefit of using top singular basis of pre-trained weights compared to middle and bottom basis or simply using random initialization is empirically demonstrated by extensive ablation studies.

### Strengths
Unlike AdaLoRA that uses SVD to dynamically adjust the rank of each adaptation matrices, LoRA-XS simply uses the top-r singular vectors  from SVD of pre-trained weights to construct learnable $r$ x $r$ matrices. The approach is quite simple and straightforward yet accomplishes the following contributions. 

1. Improve parameter efficiency while maintaining the model performance

Unlike standard LoRA, where trainable parameters scale with the model's hidden dimension, LoRA-XS’s adaptation matrix is $r$ × $r$, meaning the trainable parameter count is fixed at $r^{2}$, independent of the model’s width. This design allows efficient fine-tuning of large language models with minimal additional memory. This $r$ × $r$ matrix effectively captures task-specific adjustments in the weight space, leveraging the important information encoded in top-singular basis and maintains competitive performance with fewer parameters.

2. Extensive ablation studies to empirically justify the design rationale

The paper includes comprehensive ablation studies to validate the importance of using top-r singular vectors. They compare the model performance with varying subspaces (top, middle, bottom singular vectors), rank fractions, and initialization methods to understand the impact of basis choice on model performance. Results consistently show that the top-r singular vectors retain the most task-relevant information (except intermediate.dense layer, which seems heavily affected by the input data), aligning well with the directions already present in the pre-trained weights. This finding demonstrates that LoRA-XS can maintain competitive performance by focusing only on these crucial components, while significantly reducing parameter usage.

### Weaknesses
1. Computational cost for SVD for each pre-trained weight matrix. 

Since LoRA-XS freezes the orthogonal matrices for pre-trained $W$, they need to fully compute the SVD instead of approximating them via regularization techniques as AdaLoRA did. The computational load for computing SVD for every and each $W$ would be very heavy as the complexity for SVD is $O(min(d_1, d_2)d_1d_2)$. My concern is that the initial cost from computing SVD might offset the parameter efficiency gains during the adaptation phase, especially for LLMs with large model width. 
 
2. Ambiguous theoretical role of the $r$ x  $r$ matrix.

Besides of adding flexibility to frozen orthogonal top-r singular vectors, what's the theoretical role of $r$ x  $r$ matrix? The paper briefly mentions about the slight shift in gradient distribution, but can't see any clear explanation about the theoretical relation between this $r$ x  $r$ matrix and distribution shift to adjust to the fine-tuning data.  

3. A minor concern : The approach lacks novelty. 

The simplicity of LoRA-XS can be seen as both a strength and a limitation. On one hand, the approach lacks conceptual novelty, as other LoRA-based methods have similarly employed SVD to capture core singular components from pre-trained weight matrices. Simply extracting the top-r singular vectors and freezing them with additional $r$ x $r$ matrix for task-specific tuning seems too heuristic. However, LoRA-XS still provides strong empirical evidences to justify its choice of basis, demonstrating that focusing on the top singular components can achieve competitive performance with high parameter efficiency.

### Questions
Computational Cost for SVD: Have you considered alternative techniques to reduce the computational burden of performing full SVD on each weight matrix? 

Theoretical Role of $r$ x $r$ Matrix: Can you clarify the theoretical role of the $r$ x $r$ matrix in shifting the weight distribution to adapt to fine-tuning data?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a LoRA-XS method for LLM PEFT with extremely small trainable parameters. The authors evaluate various downstream tasks to show the good performance compared to other baselines.

### Strengths
1. The authors propose LoRA-XS, which uses only extremely small trainable parameters for PEFT scenarios.

### Weaknesses
1. The proposed method aims to learn the tra
2. The results of full parameter fine-tuning for Mixtral and Gemma are directly taken from other resources, and the performances are lower than LoRA FT, which is a little bit strange.
3. LoRA fine-tuning results for LlaMA series are also directly taken from other resources.
4. LLM with LoRA-XS requires more space to store U, V, and Σr, and requires one more matrix multiplication compared to LoRA.

### Questions
I am concerned about the evaluation part. Directly taking the results from other papers is not the wrong choice, but it could be better to evaluate all baselines by the authors themselves so that the evaluation could be fair enough by running on the same platform and controlling all hyper-parameters. It is because different platforms (e.g., different CUDA versions, different torch, transformers versions, AMP training, whether using BF16, etc) may lead to different results.

1. I know the full parameter fine-tuning may rely on too much computational resource, could the authors do their best efforts to do LoRA fine-tuning on LlaMA 2-7B and LlaMA3-8B evaluation?
2. Directly doing SVD on weight matrices usually shows unignorable errors; could the author explain why the proposed method (i.e., applying a weight W) could reduce this error?
3. Could the author do the evaluation for efficiency in terms of both HBM costs and the computational consumption by applying one more Matmul?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper presents LoRA-XS (Low-Rank Adaptation with eXtremely Small number of parameters), a novel low-rank adaptation method for large language models. As the growth of large language models highlights the need for parameter-efficient fine-tuning methods, LoRA-XS is introduced to address the storage challenges of serving multiple LoRA modules. By inserting a small trainable weight matrix between frozen low-rank matrices constructed via Singular Value Decomposition (SVD), LoRA-XS significantly reduces trainable parameters while showing competitive or better performance. Evaluations across various benchmarks demonstrate its effectiveness and an ablation study provides insights into its underlying mechanisms. LoRA-XS offers a more efficient path for model personalization and task-specific optimization.

### Strengths
* LoRA-XS reduces the number of trainable parameters by over 100x in large-scale models without sacrificing performance, enabling the deployment of millions of personalized models with minimal memory overhead.
* LoRA-XS allows for precise control of the number of additional parameters and is independent of model dimensions, providing flexibility in memory usage and being more storage-friendly and adaptable.
* LoRA-XS outperforms LoRA and other recent methods like VeRA across various model sizes and a wide range of tasks while retaining the advantages of LoRA such as no architectural modifications and no additional inference latency.

### Weaknesses
* The discussion on related works such as including [1-2] can be further improved, given the rapid progress on LoRA-based parameter efficient fine-tuning of LLMs.
* The representation ability of LoRA-XS seems to be weaker than LoRA since the space resulting from LoRA-XS is much smaller than LoRA regarding the dimension. Could this make LoRA-XS easier to overfit than LoRA? Besides, is it possible for LoRA-XS to be harder to learn on more difficult tasks than LoRA?
* Cound the authors showcase the results on LLMs larger than 7B to further demonstrate the effectiveness of the LoRA-XS.
* The proposed Theorem 3.1 is rather general and only related to the derivation of LoRA-XS. Therefore, it feels a little overclaimed by writing it as a Theorem since the theoretical contributions are limited.

[1] QA-LoRA: Quantization-Aware Low-Rank Adaptation of Large Language Models, ICLR 2024

[2] Parameter Efficient Fine-Tuning with Discrete Fourier Transform, ICML 2024

### Questions
Please kindly refer to the Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 6

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
The paper proposes a extremely parameter-efficient low-rank adaptation methodology, called LoRA-XS. It uses Truncated SVD on the weight matrices of a pre-trained Transformer model to fix the adapter matrices A and B, and only learns a small $r \times r$ matrix between them. This allows the remaining modules to remain frozen, drastically reducing the number of learnable parameters while still achieving competitive performance compared to existing models. Experimental results show competitive performance on multiple datasets, including GLUE, GSM8K, and MATH.

### Strengths
- Experiments have been conducted on various domains, and sufficient ablation studies have been performed on the introduced modules.
- LoRA-XS can achieve similar performance than LoRA with significantly fewer trainable parameters, which greatly reduces learnable parameters.

### Weaknesses
W1. Since the input is mapped to a subspace of $W$ and adjusted in scale within that space, if the distribution of the dataset used for fine-tuning differs significantly from that of the pre-training dataset, it may not adapt adequately.

W2. In Table 3, although the computational efficiency is promising since the learnable parameters are drastically reduced, the performance decrease is noticeable. For instance, in the case of MATH dataset in Gemma model, the performance decreases from 31.28 when LoRA is used to 27.62 when LoRA-XS is applied. It is questionable whether this performance can be said to be comparable to the parameter efficiency advantage.

W3. In line 280, the authors refer to “(to $r$ eigenvalues).” However, in general, the pre-trained weights $W$ may not always be applicable for eigendecomposition, meaning that eigenvalues may not always exist.

(minor typo) I think in 311 line, $h=xW + x\Delta W=xW+xARB$ should be corrected as $h=Wx+\Delta Wx=Wx + ARBx$.

### Questions
Q1. The results reported in Table 1 exclude QQP and MNLI datasets from the GLUE task. As far as I know, these two datasets are significantly larger than other reported datasets. Is there any reason why you did not conduct experiments on these two datasets? Personally, I think it is because larger datasets are complex and require more ranks and parameters. Can you conduct additional experiments on these datasets?

Q2. Although the learnable parameters are greatly reduced, the matrix multiplication computation still seems similar to LoRA. I am wondering about the results for the actual runtime/GPU peak usage of the model.

### Soundness
3

### Presentation
3

### Contribution
2

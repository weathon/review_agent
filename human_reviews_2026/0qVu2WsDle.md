# Toward Bit-Efficient Dataset Condensation: A General Framework

- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Dataset condensation aims to distill a large-scale dataset into a compact set of synthetic samples for efficient training. Existing methods primarily focus on reducing the number of samples but generally assume full-precision representations. While effective, this assumption limits their applicability in resource-constrained scenarios due to several major drawbacks:
(1) \textit{Transmission bottlenecks}—full-precision datasets consume excessive bandwidth and introduce latency during network transfer, especially in cloud–edge collaborative learning;
(2) \textit{Memory overhead}—storing and processing full-precision data rapidly exhausts GPU memory or RAM, restricting batch sizes; and
(3) \textit{Hardware underutilization}—modern accelerators are optimized for low-precision operations, yet full-precision data prevents full efficiency gains in training and inference.
To address these challenges, we propose a novel approach that fine-tunes distilled full-precision datasets into compact low-bit representations, substantially reducing memory usage with minimal computational overhead. Central to our method is a differentiable bit-conscious optimization framework. This framework allows more synthetic samples to be stored within the same memory budget, thereby improving downstream performance.
Beyond the algorithmic contribution, we provide theoretical analysis that characterizes (1) the trade-off between compression error and generalization error under memory constraints, and (2) the extent to which Fisher information is preserved under bit compression. Extensive experiments on multiple benchmarks against state-of-the-art baselines validate both the effectiveness and efficiency of our method.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper looks at "dataset condensation," which is the task of shrinking a giant dataset (like ImageNet) into a tiny set of synthetic images that can be used to train a model just as well. The authors' key insight is that everyone has been focused on shrinking the number of images (e.g., 10 images per class) but has ignored the fact that these images are still stored in full 32-bit precision, which makes them huge in terms of file size. This creates real-world bottlenecks for sending data from the cloud to an edge device or in federated learning.
The paper proposes a general framework to learn a small set of quantized (i.e., low bit-rate) synthetic samples from the start.

### Strengths
1.	I find this paper to be original and well-motivated. The authors identified a super practical, high-impact problem that the rest of the field seems to have overlooked. The "bits-per-image" dimension is just as important as the "images-per-class" dimension for any real-world deployment, and this paper is one of the first to tackle it head-on.

2.	The significance here is obvious. A method that produces a dataset that is both small in number and small in file size is exponentially more useful than the current state-of-the-art. The proposed method can be easily combined with other state-of-the-art dataset condensation methods.

3.	The theoretical analysis shows the upper bound of errors under fixed bits.

### Weaknesses
1.	Even though the idea of focusing on the total memory size of the condensed dataset is interesting, the technical contribution is relatively low. The quantization-aware training is not new but well-explored. 

2.	Some experiments are missing in Table 3 compared to Table 2.

3.	The Comparison with standard image compression techniques is not clear. Could the authors provide more details such as how and where JPEG is applied to?

4.	When the total memory is fixed, it will be interesting to see how the changes of # of images and bits per pixel affect the final accuracy. Could the authors provide some results regarding this?

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper presents a Bit-Conscious Dataset Condensation (BCDC) framework that integrates a Differentiable Data Quantization (DDQ) module to enable low-bit dataset condensation. The method allows joint optimization of data content and quantized representation, and provides theoretical analysis on the trade-off between quantization and generalization errors under a fixed memory budget. Experimental results on ImageNet-1K, cross-architecture, and continual learning settings demonstrate consistent performance gains within the same memory constraints.

### Strengths
1. This paper is the first to explore bit-efficient dataset condensation, addressing a key bottleneck of DC methods in low-resource environments.
2. The introduction of the DDQ module effectively overcomes the non-differentiability of hard quantization through smooth approximation, enabling joint optimization of condensed content and bit representation.
3. The paper provides a rigorous theoretical analysis that establishes a clear trade-off between quantization error and generalization error under memory constraints.

### Weaknesses
1. The motivation requires clarification. The paper should more clearly explain the three challenges: bandwidth cost, memory consumption, and hardware efficiency, since dataset condensation itself already reduces these costs. It remains unclear why further quantization is necessary.
2. The discussion of related work is insufficient. The authors should include prior studies connecting quantization and dataset condensation, as this directly motivates the proposed approach of enhancing DC via quantization.
3. The paper mentions that the proposed method fine-tunes full-precision condensed datasets. However, during BCDC optimization, the full-precision synthetic data $\mathcal{S}$ still needs to be stored in memory. Considering that memory efficiency is a key motivation, this design limits memory savings during training, with benefits only evident in the final quantized dataset $\mathcal{S}_{quant}$. The paper should clarify the actual memory impact during optimization.
4. Experimental results show that performance peaks at 4-bit precision and declines thereafter, mainly due to the reduced number of stored samples. Although the theoretical section discusses the trade-off, the main paper lacks in-depth analysis of the performance degradation beyond the optimal bit width, which should be further explored.
5. The DDQ module introduces parameters $k$ and $s$. The paper lacks a sensitivity analysis and practical guidance for choosing the key smoothing parameter $k$. An inappropriate value of $k$ may lead to inaccurate approximation or unstable training.

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
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a novel and interesting problem to the dataset condensation community: optimizing condensed datasets for bit-efficiency. The authors propose the Bit-Conscious Dataset Condensation (BCDC) framework, which integrates a differentiable data quantization (DDQ) mechanism into the optimization loop. The core hypothesis is that under a fixed memory budget, storing a larger number of low-fidelity (low-bit) synthetic samples is more effective for training a downstream model than storing a smaller number of full-precision samples. The paper provides theoretical analysis and extensive experiments to support this claim, showing significant performance improvements over state-of-the-art methods when memory constraints are enforced.

### Strengths
- The paper's primary strength is its originality. It shifts the paradigm of dataset condensation from merely minimizing the number of samples to optimizing the total information density within a given bit budget.
-  The experiments convincingly demonstrate the effectiveness of BCDC under the "Same Memory" (SM) condition. The performance gains shown in Tables 3, 4, 5, and 6 are substantial and provide strong evidence for the paper's central hypothesis on the tested datasets. The ablation study in Table 8 successfully highlights the importance of the joint optimization approach over a naive post-hoc quantization baseline.

### Weaknesses
- The paper claims a "modest increase in training cost—ranging from 17% to 20%" (Sec 5.2, Table 9). This analysis is insufficient and potentially misleading. The complexity of the DDQ process, particularly the backpropagation through the  `tanh`  approximation and the additional quantization loss term, is not constant. The paper provides no analysis of how this overhead scales with data dimensionality (e.g., higher-resolution images) or the number of quantization levels (i.e., bit-depth). A 20% overhead on CIFAR-10 could easily become a prohibitive bottleneck on more complex, real-world data, which challenges the method's claimed efficiency.
- Is this trade-off of per-sample fidelity favorable? For tasks that depend heavily on high-frequency information, such as fine-grained classification (e.g., distinguishing bird species) or medical image analysis where subtle textural anomalies are crucial, this specific form of information loss could be disastrous. The paper's claims of general improvement are unsubstantiated because the "richer information" from more samples may not be the right information for all tasks.
- The BCDC framework introduces new hyperparameters, most notably the quantization loss weight \lambda  and the  tanh  sharpness parameter  k . The main paper fixes \lambda = 0.2  and relegates a brief sensitivity discussion to the appendix, which is insufficient for a core component of the optimization objective. The method's performance likely hinges on a careful balancing of the validation and quantization losses, governed by $\lambda$ .

### Questions
Please refer weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper explores the problems of dataset distillation in resource-constrained conditions. Distilled datasets typically adopt full-precision representations. However, under resource-constrained condition, it will lead to transmission bottlenecks, excessive memory overhead, and insufficient hardware utilization issues. This paper proposes BCDC framework, applying low-bit quantization to achieve efficient data storage and training in resource-constrained conditions. It can be applied to the existing dataset distillation method. Specifically, BCDC introduces a differentiable bit-aware optimization algorithm to fine-tune the distilled full-precision data into a low-bit representation, allowing for the storage of more synthetic data within the same memory budget. This paper also theoretically analyzes the trade-off between quantization error and generalization error, and also the preservation of Fisher information. BCDC can be applied to existing dataset distillation method with a slight performance drop.

### Strengths
1. This paper presents a clear problem and motivation. It tackles the limitations of current dataset distillation in resource-constrained settings by introducing low-bit quantization to improve both storage and computational efficiency.
2. This paper provides a detailed theoretical analysis of the trade-off between quantization error and generalization error under fixed memory constraints, and gives the limits for Fisher information preservation under bit compression.
3. This paper is well-written with clear presentation.

### Weaknesses
1. As the proposed method adopt low-bit quantization to reduce the storage requirements for distilled images, more samples can be stored with the same memory budget. It is known that more training distilled samples will improve downstream performance under dataset distillation settings. However, more training samples also increase downstream training time, which actually contradicts to training efficiency purpose of dataset distillation.
2. This paper does not explicitly state whether the quantized synthetic data is processed in low-bit form during downstream training. From the description, differentiability appears to be primarily used as a soft approximation in the optimization process, but how it is processed during downstream training is not stated in detail.
3. This paper claims that distilled datasets in low-bit formats can improve hardware utilization, but no empirical evidence or explanation is provided to support the claims.
4. This paper only compares the performance with full-precision dataset distillation methods, neglecting distilled data parameterization methods. Distilled data parameterization methods store the distilled data in a low-dimensional or implicit form, which can also reduce storage costs. This weakens the fairness of the evaluation and makes it difficult to demonstrate the effectiveness of the proposed method.

### Questions
1. Given the same downstream training time budget, how do storage and downstream performance compared with the full-precision method?
2. It would be helpful if the authors could provide implementation details for downstream training and how the proposed method improves hardware utilization.
3. The authors could consider briefly discussing how their proposed method differs from data parameterization dataset distillation methods based on latent variables, and may include them in a baseline comparison.
4. Can this method be used for the distilled data generated from other tasks?

### Soundness
3

### Presentation
3

### Contribution
2

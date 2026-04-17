# Efficient Orthogonal Fine-Tuning with Principal Subspace Adaptation

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 6

## Abstract
Driven by the rapid growth of model parameters, parameter-efficient fine-tuning (PEFT) has become essential for adapting large models to diverse downstream tasks under constrained computational resources. Within this paradigm, orthogonal fine-tuning and its variants preserve semantic representations of pre-trained models, but struggle to achieve both expressiveness and efficiency in terms of parameter counts, memory, and computation. To overcome this limitation, we propose efficient Orthogonal Fine-Tuning with Principal Subspace adaptation (PSOFT), which confines orthogonal transformations to the principal subspace of pre-trained weights. Specifically, PSOFT constructs this subspace via matrix decomposition to enable compatible transformations with higher rank, establishes a theoretical condition that strictly maintains the geometry of this subspace for essential semantic preservation, and introduces efficient tunable vectors that gradually relax orthogonality during training to enhance adaptability. Extensive experiments on 35 NLP and CV tasks across four representative models demonstrate that PSOFT offers a practical and scalable solution to simultaneously achieve semantic preservation, expressiveness, and multi-dimensional efficiency in PEFT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes PSOFT (Principal Subspace Orthogonal Fine-Tuning), a novel and efficient orthogonal fine-tuning framework for large pretrained models. By constraining orthogonal transformations within the principal subspace of pretrained weights, PSOFT effectively balances geometric preservation, expressive capacity, and computational efficiency. The method employs SVD-based subspace extraction, Cayley parameterization for strict orthogonality, and learnable scaling factors for task adaptability.

### Strengths
1. PSOFT introduces an elegant combination of orthogonal fine-tuning and low-rank adaptation through principal subspace projection, offering a fresh perspective on improving scaling efficiency.
2. The paper provides a clear mathematical justification for subspace-constrained orthogonality and uses Cayley parameterization to guarantee exact geometric preservation.
3.  Extensive experiments across diverse NLP and CV tasks show consistent improvements over LoRA and BOFT, with significantly lower memory consumption and stable performance on large models.

### Weaknesses
1.  The paper could analyze more systematically how the choice of principal subspace rank affects performance and efficiency.
2.  While results on up to 8B models are shown, the paper lacks discussion on potential limitations or stability when scaling beyond that range.
3. Are the principal subspaces learned independently for each layer, or do they share correlated bases across layers? If the latter, does PSOFT exploit inter-layer subspace alignment to further improve transferability?
4. Beyond downstream accuracy, have the authors analyzed how PSOFT affects representation geometry—for instance, by measuring canonical correlation alignment (CCA) or subspace angles before and after fine-tuning?
5. Orthogonal fine-tuning implicitly constrains updates on the Stiefel manifold. How does PSOFT’s subspace constraint modify the tangent space of optimization compared to full-space OFT, and what implications does it have for curvature and convergence speed?

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces PSOFT, a PEFT algorithm for pretrained models, incorporating orthogonal fine-tuning strategy into additive fine-tuning framworks. Extensive experiments on several tasks validates the effectiveness of this method.

### Strengths
1. This paper rethinks the updates of fine-tuning within the principal space of the original weight matrix.

2. The method demonstrates strong results, consistently outperforming LoRA, PiSSA, and other OFT variants (BOFT, GOFT) on a wide range of benchmarks, including GLUE, VTAB-1K, GSM-8K, and commonsense reasoning.

3. PSOFT achieves its strong performance while using significantly fewer parameters than competitors. In Table 2, PSOFT and GOFT (0.08M) achieves the best average performance while being much more parameter-efficient than BOFT (1.41M) and LoRA (1.33M).

### Weaknesses
1. A misclaim of the core idea of the proposed algorithm: The key idea of "orthogonal fine-tuning" is keeping correlations between neurons (say inner products) unchanged after fine-tuning, thus providing semantic-preservation. However, this does not hold for PSOFT. In fact, PSOFT is actually an additive fine-tuning method similar to LoRA, $\Delta W = W_{final} - W_{pre} = (A'RB' + W_{res}) - (A'B' + W_{res}) = A'(R-I)B'$, where the updates are restricted to principal subspace and enjoy a natural low-rank property. Yet, $(W_{pre} + \Delta W)^\top (W_{pre} + \Delta W)$ is not equal to $W_{pre} ^ \top W_{pre}$. Therefore, it is not correct to claim that this is an orthogonal fine-tuning method, and does not achieve the so-called semantic-preservation in Intro.

2. What has PSOFT done?: Keep the orthogonal basis of the original pretrained weights unchanged, and adjust the main ranks of the entire space by adjusting the coordinates using orthogonal transformation. So the expressiveness is actually bounded by the identical orthogonal bases of the pretrained weights. Is this expressive enough? It seems that LoRA could span a larger subspace of the updates.

3. Some doubts of the experimental results: The experiment results seem to show great improvements. But the OOM phenomena of GOFT seems weird and not convincing, as they incorporate only vector hadamard multiplications on activations to implement their forward, this also means the backward will not consume large matrix storage in GPUs. Moreover, BOFT incorporates matrix multiplications instead, and I think GOFT should be more memory-efficient than BOFT. I suggest conducting analyses on memory consumptions of their forward and backwards of a single layer through rigorous calculations and deductions (better with experimental evidence) to support your results. At the very least, as you indeed have NVIDIA H100 80GB for decoder experiments, why not conduct the OOM experiments on those chips?

### Questions
See Weaknesses.

Some further questions:
1. Following on Weakness 2, is the original principal subspace always sufficient for downstream tasks? Can we reach a conclusion that the principal space is a better regularization of the entire low-rank subspaces? Is it possible that the diminishing returns at higher ranks (Table 17) are a symptom of this fixed-subspace limitation?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents PSOFT, a novel parameter-efficient fine-tuning (PEFT) method that confines orthogonal fine-tuning to the principal subspace of pre-trained weights, identified via SVD. The goal is to merge the efficiency of low-rank adaptation (LoRA) with the semantic preservation of orthogonal fine-tuning (OFT). The method demonstrates a strong balance of accuracy, parameter efficiency, and memory usage across a wide range of NLP and CV tasks, often outperforming existing methods.

### Strengths
* The core idea of applying orthogonal transformations within a low-rank principal subspace is an intuitive and effective way to bridge the gap between LoRA and OFT.
* The comprehensive evaluation across 35 NLP and CV tasks shows strong performance. PSOFT is not only accurate but also highly parameter- and memory-efficient, crucially avoiding the out-of-memory (OOM) errors that plague other OFT variants.
* The paper successfully highlights that efficiency is multi-dimensional. PSOFT shows clear advantages in memory footprint and training speed, making it a practical and scalable solution for large models.

### Weaknesses
1.  **Theory vs. Practice:** The paper emphasizes "strict" geometry preservation as a key benefit, yet the best-performing algorithm intentionally relaxes this condition with tunable vectors to improve results. Could you discuss the trade-off here and the impact of this relaxation on the semantic preservation you aim for?
2.  **"Effective Rank" Definition:** The claim of a "higher effective rank" is based on a non-standard definition (`r_PSOFT = √M`). This is confusing and weakens the claim of higher expressiveness. Could you please clarify this definition and provide a more rigorous justification?
3.  **Guidance on Hyperparameter `r`:** The method's performance is tied to the choice of rank `r`, but the paper lacks an analysis of its sensitivity or guidance on how to select it for different tasks or models.
4.  **Implementation Clarity:**
    *   The choice of K=5 for the Neumann series approximation is stated without justification. What is the impact of this choice on orthogonality and training speed?
    *   Reporting performance in relative speedup ("2.1x") without absolute wall-clock times makes it difficult to assess the true computational overhead. Could you provide absolute training time comparisons?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces PSOFT, a parameter-efficient finetuning (PEFT) approach that combines the low-rank decomposition and orthogonality principles from LoRA and OFT-style methods. Specifically, PSOFT first applies an SVD decomposition to the pretrained weight matrices, and then inserts a learnable rotation matrix between the decomposed components. The SVD decomposition enforces low-rank structure and orthogonality, while the rotation matrix provides flexibility for task-specific adaptation. Experimental results on both encoder-only and decoder-only models across multiple benchmarks demonstrate the effectiveness of PSOFT.

### Strengths
The idea is conceptually elegant and well-motivated. Combining SVD-based low-rank decomposition with a learnable rotation matrix is a natural way to balance efficiency and expressiveness. The orthogonal projection helps preserve the representational capacity of the subspace, while the rotation matrix allows fine-grained adaptation to downstream tasks.

The paper is clearly written and easy to follow. The method is presented in a straightforward manner with comprehensive experimental validation across various settings.

### Weaknesses
1. The related work section lacks a clear comparison and differentiation between PSOFT and closely related methods such as **DoRA** and **LoRA-XS**. A more detailed discussion highlighting conceptual and empirical differences would strengthen the contribution.

2. The reported ranks $r$ (e.g., 46, 354, 424) are irregular and seem to vary considerably. This suggests possible **sensitivity to hyperparameter tuning**, which should be verified through an ablation study on $r$.

3. It appears that all results are produced via **re-implementations** rather than directly using reported numbers from prior works. While this ensures consistency, it would also be useful to include **direct comparisons under standardized settings** from previously published benchmarks to contextualize the gains.

### Questions
The number of tunable parameters changes significantly from 0.08M to 12.2M while other methods like OFT do not change so much (from 1.4M to 11.6M).
Whether it stems from differences in model architecture, rank choice, or implementation details?

### Soundness
2

### Presentation
3

### Contribution
3

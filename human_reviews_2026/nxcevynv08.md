# Thicker and Quicker: The Jumbo Token for Fast Plain Vision Transformers

- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
ViTs are general and accurate, and address many tasks, but ViTs are slow, and are not always practical when efficiency is key. Existing methods for faster ViTs design hybrid non-ViT architectures, losing generality, or shrink their tokens, sacrificing accuracy. Many non-ViT architectures are both fast and accurate. Yet, without significant modifications, they cannot do what ViTs can: process other input shapes, pre-train by SOTA self-supervised learning, reduce computation by dropping tokens, and more. We make ViTs faster by reducing patch token width while increasing global token width by adding a new Jumbo token. Our wider Jumbo token is processed by its own wider FFN to increase model capacity. Yet our Jumbo FFN is efficient: it processes a single token, for speed, and its parameters are shared across all layers, for memory. Crucially, our Jumbo is attention-only and non-hierarchical, like a plain ViT, so it is simple, scalable, flexible, and compatible with ViT methods new and old. Jumbo improves over ViT baselines with Registers from Nano to Large scales while maintaining speed/throughput on ImageNet-1K (0.1-13%). Jumbo also improves segmentation (1.9-3.1% on ADE20K), MAE pre-training (4.9% linear probing on ImageNet-1K), test-time adaptation (5.2% on ImageNet-C), and time series modeling. Our Jumbo models even achieve better speed-accuracy trade-offs than specialized non-ViT compute-efficient models, while maintaining plain-ViT compatibility for practicality. Code and weights are available: https://github.com/antofuller/jumbo

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces Jumbo, a simple yet effective extension of Vision Transformers (ViTs) that replaces the conventional single CLS token with a wider one. The “Jumbo” token is divided before self-attention and then reassembled afterward, enabling more powerful global representation learning at minimal additional cost. The approach preserves the elegance and simplicity of standard ViTs while integrating seamlessly with techniques such as token dropping and multimodal learning. Extensive experiments demonstrate consistent performance gains across various benchmarks, including ImageNet-1K and ImageNet-21K.

### Strengths
The Jumbo design can be seamlessly integrated into existing ViT architectures without requiring substantial structural modifications. It consistently outperforms both ViT+Registers and several optimized models, with particularly strong gains observed in smaller model settings. The paper provides extensive empirical evidence, including comprehensive comparisons, ablation studies, and diverse use-case evaluations.

### Weaknesses
Compared with ViT+Registers, the primary difference lies in using a single large token instead of multiple smaller ones—a design choice that may appear incremental rather than fundamentally innovative. Although Jumbo achieves notable performance gains, it comes at the cost of increased computation and a larger number of parameters, which may limit its practicality in real-world applications. The contributions are incremental.

### Questions
The proposed approach can potentially have a big increase in the cost of efficiency with a little performance gain.

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes ViT+Jumbo which is a attention-only and non-hierarchical design that increases global capacity by replacing the CLS token with a single Jumbo token. To increase the model capacity, the Jumbo token is wider than patch tokens and has its own FFN whose parameters are shared across layers. The Jumbo token is split before MHSA and re-concatenated after attention. Only the Jumbo token passes through the Jumbo-FFN, keeping overhead small. The authors provide extensive experiments results to verfr their modification design on ImageNet-1K/21K, MAE pretraining, test-time adaptation, and time-series classification. Experimental results suggest that ViT+Jumbo improves speed-accuracy trade-offs versus ViT baselines and competes with specialized efficient backbones while preserving ViT compatibility.

### Strengths
1. Good practicality: The method remains attention-only and non-hierarchical, then the Jumbo token simply widens global processing while keeping the ViT interface intact. This preserves applicability of token dropping, SSL, and ViT heads. The design requires only two splits and concats plus a shared FFN with minimizing implementation burden while adding capacity where it matters.
2. Increasing capacity while maintaining throughput is promising: On ImageNet-1K the method attains or approaches the frontier across speeds and outperforming register-base models especially at small sizes. This shows effectiveness under constrained regimes.
3. This paper tests the proposed scheme from different experimental perspectives and tasks (supervised classification, SSL, TTA, and time-series classification), and provides a wealth of ablation experimental results and a relatively detailed explanation of the effects of hyperparameter tuning, which is commendable.
4. Some visual comparison results are provided, especially the visualization of the norm range, which can be compared according to the assumption of its most relevant baseline (i.e., the Register method).

### Weaknesses
1. Fairness and measurement choices may bias comparisons: ImageNet-1K training uses DeiT-III distillation and a specific two-stage recipe. But it may unclear that all baselines were tuned equivalently under this recipe. Without such distillation training, could the proposed schedule works well to coverage? 
2. No experiments for downstream tasks like the common compared detection and segmentation which are important for verifying the robustness of vision encoders. So downstream dense tasks remain unverified as vision processing is the main field of the work. Besides, how to utilize Jumbo tokens at different layers in different downstream tasks is also worth exploring.
3. In the MAE self-supervised learning experiment, the linear probing setting was used to illustrate that performance cannot fully reflect the model's capabilities. Full fine-tuning is masking SSL, which is a more common choice, including MAE. It is better to add such results.

### Questions
1. Why "While many non-ViT architectures are both fast and accurate, they cannot flexibly process other input shapes" in the abstract? Non-hierarchical non-ViT models such as Vim and V-RWKV are also sequence models. How do they differ from ViTs in terms of input shape?
2. How do register-based methods compare to Jumbo in terms of performance, given similar model capacities?
3. Why does the Jumbo method work more effectively on smaller models? Can it be understood that the benefits are more pronounced with larger model sizes?
4. Please refer to the section on Weaknesses for other questions.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes adding a jumbo token to ViTs, demonstrating good efficiency compared to ViT register baselines. The paper is well-written and the idea is intuitive.

### Strengths
1. The paper is well-written.
2. The authors perform systematic and high-quality ablation studies.

### Weaknesses
1. Thanks for the authors' discussions in Sec. 4.7, which show that the additional parameters indeed improve the accuracy. The advantage of Jumbo token is the efficiency, which limits the upper bound of this method.

2. The current use case seems limited. I am curious to see its potential in DiT for C2I or T2I image generation. 

3. As discussed in the limitation section, does this method have potential in visual-language understanding or language models?

4. Can the authors explain and analyze why Jumbo reduces the high norms of tokens more effectively than Register as shown in Fig. 5?

### Questions
1. Can the authors report the model parameters in Tab. 5 for the Alt.1 and Alt. 2 model variants?
2. Why did the authors perform ablations on different scales of ViTs in Tab. 5? Would the conclusions change for Alt. 1 and Alt. 2 if using ViT-Base models?

### Soundness
3

### Presentation
3

### Contribution
3

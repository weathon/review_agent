# Study of Training Dynamics for Memory-Constrained Fine-Tuning

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 6, 8, 4, 6

## Abstract
Memory-efficient training of deep neural networks has become increasingly important as models grow larger while deployment environments impose strict resource constraints. We propose TraDy, a novel transfer learning scheme leveraging two key insights: layer importance for updates is architecture-dependent and determinable a priori, while dynamic stochastic channel selection provides superior gradient approximation compared to static approaches. We introduce a dynamic channel selection approach that stochastically resamples channels between epochs within preselected layers. Extensive experiments demonstrate TraDy achieves state-of-the-art performance across various downstream tasks and architectures while maintaining strict memory constraints, achieving up to 99\% activation sparsity, 95\% weight derivative sparsity, and 97\% reduction in FLOPs for weight derivative computation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes TraDy (Training Dynamics), a memory-efficient fine-tuning approach for deep neural networks under strict resource constraints. The method leverages two key insights: (1) layer importance for updates is architecture-dependent and can be determined a priori, and (2) dynamic stochastic channel selection provides superior gradient approximation compared to static approaches. The authors introduce a Reweighted Gradient Norm (RGN) metric that accounts for both gradient magnitude and memory costs. TraDy operates by pre-selecting important layers based on architectural properties, then dynamically resampling input channels between epochs within these layers. Experiments on CNNs and transformers across multiple vision and NLP tasks demonstrate that TraDy achieves competitive performance while maintaining up to 99% activation sparsity, 95% weight derivative sparsity, and 97% reduction in FLOPs for weight derivative computation.

### Strengths
1. The core idea is both creative and elegant. The central challenge of memory-constrained training is that one cannot afford to compute the very importance metrics needed to decide what to update. TraDy's solution—decoupling the problem into a static, a priori layer selection and a dynamic, stochastic channel selection—is a novel and highly effective way to break this circular dependency.

2. The authors compare TraDy against a comprehensive set of baselines, including static/dynamic and random/deterministic variants, across 3 CNN architectures, 7 vision datasets, 3 memory budgets, and 3 random seeds.

3. Well-structured paper with clear problem formulation and notation.

### Weaknesses
1. The main comparison is against Sparse Update (SU). More recent and relevant works are mentioned but not included in the main experiments. For example, how is the proposed method compared with "SMT: Fine-Tuing Large Language Models with Sparse Matrices".

2. LoRA and other parameter-efficient fine-tuning methods are not discussed or compared.

3. The authors acknowledge this limitation explicitly. The paper lacks actual on-device experiments showing latency, energy consumption, or throughput on target edge devices.

4. The paper is compactly formatted and the template style is changed.

### Questions
see weakness.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper propose a dynamic training method (TraDy) for memory-constrained scenarios. . By stochastically resampling channels between epochs within architecturally important layers, the experiments demonstrate the effectiveness of the proposed method across various transfer learning scenarios. This article is well-written and well-organized.

### Strengths
1. This paper is grounded in a solid theoretical foundation, featuring detailed derivations and rigorous argumentation.
2. This article is well-written and well-organized.

### Weaknesses
1.Fig. 2 is positioned too far from the corresponding text section. It is recommended to optimize the image layout.
2.The baseline for existing studies compared in this paper is limited, with only SU available.
3.It is recommended to highlight the best-performing results in Tables 1-5.

### Questions
1.Although the author claims that lennec et al.’s implementation excludes activation memory from their budget calculations, it doesn't seem to affect adding it as an additional baseline to Table 1-5. Why wasn't this done?

### Soundness
4

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
3

### Summary
The paper presents TraDy, a novel approach for fine-tuning pre-trained deep neural networks to downstream tasks under memory-constrained budgets. TraDy proposes a dynamic channel selection scheme that exploits the gradient sparsity of input channels to achieve both weight and activation sparsity during fine-tuning for downstream tasks.

### Strengths
- The analysis of the stochastic gradients' heavy-tailed behaviour during fine-tuning of a pre-trained network, relative importance of the network layers, consistency across downstream tasks and channel importance distribution is rigorously presented and well explained.

### Weaknesses
- The paper is currently lacking comparison with the state-of-the-art fine-tuning reported in section 2 (i.e., Lin et. al (2022), Kwon et al. (2024) and Quèllenec et al. (2024)). 
- TraDy performances are reported in the main paper only for CNN models.  
-  Unfortunately, the plots reported in Fig. 3, Fig.4 and Fig. 6 are not easy to read and to position within the main contributions of TraDy.

### Questions
- Could you maybe compare the fine-tuning to downstream tasks accuracy of TraDy against the state of the art memory constrained fine-tuning methods cited in the related works Section 2? It would be interesting to compare the computational benefits vs fine-tuned accuracy of TraDy with those of other state-of-the-art fine-tuning methods. 
- Could you maybe add the results for vision attention-based models (i.e., ViTBase), BERT and RoBERTA in the main paper? 
- How does TraDy perform when compared to low-rank fine-tuning methods (i.e., LoRA)?

### Soundness
3

### Presentation
2

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
This paper introduces TraDy, a novel hybrid approach for memory-constrained fine-tuning of pre-trained models. The core idea is to combine (1) a static, a priori layer selection process, which the authors argue is architecture-dependent, with (2) a dynamic, stochastic channel selection process, which is argued to be task-dependent.

The methodology is justified by a solid analysis of training dynamics, identifying three key insights: the heavy-tailed nature of gradients, the task-invariance of layer importance, and the task-dependence of channel importance. Experimentally, TraDy demonstrates state-of-the-art performance against static sparse update baselines (like SU) across various vision tasks and architectures, achieving significant efficiency gains, including up to 99.5% activation sparsity and 97% reduction in weight gradient FLOPs.

### Strengths
1.  **Strong Theoretical and Experimental Grounding:** The paper's primary strength lies in its thorough justification. The authors do not just propose a method but provide a robust analysis of *why* it should work. The experimental validation of the three key insights (heavy-tailed gradients, task-invariant layer ranks, task-dependent channel ranks) is convincing and provides a solid foundation for the proposed hybrid design.

2.  **High Efficiency and Strong Performance:** The method achieves impressive results, demonstrating SOTA accuracy while operating under severe memory constraints. The reported efficiency metrics (e..g., >99% activation sparsity) are highly significant for the target application of on-device learning and data-drift adaptation.

3.  **Rich Ablation Studies:** The paper's claims are well-supported by a comprehensive set of ablation studies. The comparison of `TraDy` (TopK Random) against alternatives like `Full Random` and static methods (like `S-Det RGN`) clearly isolates the benefits of both the static layer selection and the dynamic channel selection components, strengthening the paper's overall argument.

### Weaknesses
1.  **Evaluation on Simple Tasks:** The empirical evaluation, while broad in terms of datasets (CIFAR-10/100, CUB, Flowers, etc.), is primarily limited to relatively simple, small-scale classification tasks. To truly validate the robustness and scalability of TraDy, an evaluation on more complex, large-scale benchmarks (e.g., ImageNet-1K) is necessary.

2.  **Missing Comparison to Key PEFT Methods:** The paper's related work and experimental comparisons focus almost exclusively on *sparse update* methods (like SU). However, it overlooks a major and highly relevant category of Parameter-Efficient Fine-Tuning (PEFT) methods. A discussion and, ideally, a comparison against popular *adapter-based* methods that also target memory and compute efficiency would be crucial. Key missing comparisons include:
    * **LoRA** (Hu et al., 2021)
    * **DoRA** (Liu et al., 2024)
    * **PaCA** (Woo et al., 2025)

### Questions
1.  **Scalability to Large-Scale Models (LLMs):** The paper successfully demonstrates TraDy on CNNs and, in the appendix, on ViT and smaller BERT models (though with mixed results for NLP). A key question is whether this framework can be effectively scaled to the fine-tuning of modern, massive-scale models, such as LLMs with >7B parameters? How does the static layer selection and dynamic channel selection interplay in such homogeneous, transformer-heavy architectures?

2.  **Runtime Overhead of Dynamic Selection:** The paper clearly demonstrates the *memory* and *FLOPs* advantages of TraDy. However, it does not discuss the potential *wall-clock time* overhead. Does the dynamic resampling of channels at every epoch (including random number generation, index selection, and mask creation) introduce a non-negligible runtime cost compared to static methods (like SU) that perform this selection only once offline?

### Soundness
4

### Presentation
3

### Contribution
3

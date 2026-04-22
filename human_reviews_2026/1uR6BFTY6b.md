# DoRAN: Stabilizing Weight-Decomposed Low-Rank Adaptation via  Noise Injection and Auxiliary Networks

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
Parameter-efficient fine-tuning (PEFT) methods have become the standard paradigm for adapting large-scale models. Among these techniques, Weight-Decomposed Low-Rank Adaptation (DoRA) has been shown to improve both the learning capacity and training stability of the vanilla Low-Rank Adaptation (LoRA) method by explicitly decomposing pre-trained weights into magnitude and directional components. In this work, we propose **DoRAN**, a new variant of DoRA designed to further stabilize training and boost the sample efficiency of DoRA. Our approach includes two key stages: (i) injecting noise into the denominator of DoRA’s weight decomposition, which serves as an adaptive regularizer to mitigate instabilities; and (ii) replacing static low-rank matrices with auxiliary networks that generate them dynamically, enabling parameter coupling across layers and yielding better sample efficiency in both theory and practice. Comprehensive experiments on vision and language benchmarks show that DoRAN consistently outperforms LoRA, DoRA, and other PEFT baselines. These results underscore the effectiveness of combining stabilization through noise-based regularization with network-based parameter generation, offering a promising direction for robust and efficient fine-tuning of foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DoRAN, a stable and flexible extension of DoRA for parameter-efficient fine-tuning. It introduces a learnable noise term (τ) to stabilize normalization and a hypernetwork to generate low-rank matrices with cross-layer parameter sharing. Theoretical analysis shows that DoRAN improves optimization stability and sample efficiency, while experiments on both vision (ViT-B/16) and language (LLaMA-7B/13B) tasks demonstrate consistent gains over LoRA and DoRA with minimal parameter overhead.

### Strengths
1. The paper provides a clear and well-motivated improvement over DoRA by introducing a learnable noise term (τ) that stabilizes normalization and enables smooth interpolation between DoRA and LoRA behaviors. 
2. The proposed hypernetwork design allows cross-layer parameter sharing, which helps reduce redundancy to some extent and may promotes diversity in learned representations.
3. The paper includes theoretical analysis that offers useful insights into the stability and convergence behavior of the proposed method, providing a clear theoretical perspective in addition to empirical evidence.

### Weaknesses
1. The experimental results need clearer attribution. It is essential to specify which results are taken from prior work and which are reproduced by the authors. If the baseline numbers are reproduced, detailed descriptions of their experimental settings should be provided to ensure fairness and reproducibility.
2. For the image classification experiments, both the proposed method and baseline results appear to have significantly fewer data points than prior related work [1, 2, 3, 4], which raises concerns about the reliability or completeness of the evaluation.
3. The experiments should include results on more recent and competitive large language models, such as LLaMA3-8B or Qwen-2.5-7B, to demonstrate broader applicability and scalability.
4. The paper should include a comprehensive comparison of computational efficiency between DoRAN, LoRA, and DoRA.

------

[1] MLAE: Masked LoRA Experts for Visual Parameter-Efficient Fine-Tuning. https://arxiv.org/abs/2405.18897

[2] Neural Prompt Search. https://arxiv.org/abs/2206.04673

[3] Towards Efficient Visual Adaption via Structural Re-parameterization. https://arxiv.org/abs/2302.08106

[4] Sensitivity-Aware Visual Parameter-Efficient Fine-Tuning. https://arxiv.org/abs/2303.08566

### Questions
1. In Appendix B.4 (Table 5), why are experiments not conducted on all projection layers (QUERY, KEY, VALUE, UP, and DOWN) as in the original DoRA, and why is KEY ablated instead? Please report DoRAN results when applied to the full set of projections (Q/K/V/up/down) and compare them directly to the corresponding results in the original DoRA paper to ensure completeness and fairness.
2. Please provide the training dynamics and final values of the learnable noise term (τ). In particular, report per-layer (and, if applicable, per-head) trajectories over training, the final distributions, and their correlations with optimization stability and performance. A brief analysis (e.g., plots of τ over steps, summary statistics, and variance across random seeds) would substantially strengthen the paper.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes DoRAN method that extends DoRA to improve training stability and sample efficiency. The main contributions include:

1. Stabilized weight decomposition with learnable noise, mitigating DoRA’s instability.

2. Hypernetwork-based low-rank reparameterization enabling parameter coupling across layers.

3. Theoretical and empirical proof of improved sample efficiency and fine-tuning robustness.

### Strengths
1. Provide theoretic analysis on gradients to show that the learnable noise $\tau$ can ensure bounded gradients

### Weaknesses
1. Compared to other LoRA variants, one major issue with DoRA-like approaches is their high computational cost in both GPU memory and training time. The primary reason is that computing the normalization vector $\\|W_0+BA\\|_c$ requires a full matrix $BA\in \mathbb{R}^{m\times n}$, even though it has low rank as $A\in \mathbb{R}^{r \times n}$ and $B\in \mathbb{R}^{m\times r}$ with $r \ll \mathrm{min}(m,n)$. This inefficiency can be observed by conducting the natural language generation (NLG) task on the MetaMathQA dataset; see [PiSSA] for experimental details.

2. Theoretically, DoRA may suffer from instability when $\\|W_0+BA\\|_c$ becomes small, yet the authors provide no empirical evidence to support this argument. It is possible that the observed instability arises from the unbounded scaling vector $m$.

3. The motivation for parameter sharing across layers is not well justified. Even if such sharing has potential benefits, it remains unclear why a simple one-layer MLP should be used to generate $A,B$.

4. Similar to DoRA, a recent method, [DeLoRA], also introduces a normalization term: $$ y=W_0x+ B\Sigma A x$$ where $\Sigma\in \mathbb{R}^{r\times r}$ is a positive diagonal matrix which normalize the rows of $A$ and the columns of $B$. Theoretically, DeLoRA could suffer from the same instability discussed in this paper; however, it has been found empirically stable over a wide range of learning rates.

5. The paper claims that the proposed approach reduces to LoRA when $\tau$ is large. This claim is incorrect. For large $\tau$, the proposed approach becomes $$W=\frac{m}{\tau} (W_0+BA)$$ which fundamentally differs from LoRA’s formulation, $$W=W_0+\frac{\alpha}{\gamma}BA.$$ In LoRA, $BA$ often has much smaller norm than $W_0$, ensuring that pre-trained knowledge is largely preserved. In contrast, the proposed approach applies a learnable scaling to $W_0$, potentially causing a large weight shift $\Delta W:=W - W_0$.

6. The proposed method needs additional experiments to validate its effectiveness, such as NLG evaluations on the MetaMathQA dataset (as in [PiSSA]), tests on non-LLaMA models, and experiments with larger base model. 

[PiSSA] Meng et al. PiSSA: Principal singular values and singular vectors adaptation of large language models. NeurIPS 2024.

[DeLoRA] Bini et al. DeLoRA: Decoupling angles and strength in low-rank adaptation. ICLR 2025.

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes DoRAN, a Parameter-Efficient Fine-Tuning (PEFT) method that improves upon Weight-Decomposed Low-Rank Adaptation (DoRA). The authors identify two key limitations in DoRA: potential optimization instability when weight norms are small, and inefficient parameterization by learning independent low-rank matrices for each layer. DoRAN introduces two main components: (i) a learnable noise term injected into the denominator of the weight normalization step, which smoothly interpolate between DoRA's directional updates and LoRA's linear scaling; and (ii) the use of auxiliary networks (hypernetworks) to generate the low-rank adaptation matrices. This second component induces shared structure across layers and attention heads, aiming to improve parameter and sample efficiency. The authors provide a theoretical analysis, framing the method through a Mixture-of-Experts (MoE) perspective, to show that DoRAN achieves a better (polynomial) sample complexity compared to the exponential rate of non-shared structures. The method is evaluated on image classification and commonsense reasoning tasks, demonstrating performance gains over DoRA and LoRA.

### Strengths
Although the idea of interpolating between LoRA and DoRA's weight approximation is simple, it proves effective experimentally. 

Using hyper-networks to predict the weight update is also a straightforward strategy to induce correlations across layers.

The authors theoretically support DoRAN's strategy which further supports experimental findings.

### Weaknesses
Algorithmic novelty is limited as interpolating between LoRA and DoRA is a straightforward strategy and using hyper-networks for weight prediction has already been studied for PEFT [1]

Comparison with other sota PEFT algorithms is not performed. Since DoRA can theoretically be applied on top of other recent PEFT algorithms [2,3] having a study of whether the DoRAN strategy further improves these variants would have been a strong addition to the paper.

The paper only quantifies the increase in trainable parameters. There is no discussion of the practical impact on training time, memory, or FLOPs introduced by the hypernetwork's forward pass at every step. 

[1] Mahabadi, Rabeeh Karimi, et al. "Parameter-efficient multi-task fine-tuning for transformers via shared hypernetworks." Proceedings of the 59th Annual Meeting of the Association for Computational Linguistics and the 11th International Joint Conference on Natural Language Processing, 2021.

[2] Meng, Fanxu, Zhaohui Wang, and Muhan Zhang. "Pissa: Principal singular values and singular vectors adaptation of large language models." Advances in Neural Information Processing Systems, 2024.

[3] Zhang, Qingru, et al. "Adalora: Adaptive budget allocation for parameter-efficient fine-tuning." ICLR, 2023

### Questions
Hyper-network predicted weights and interpolation coefficients prove effective for vision tasks but gains due to the hypernetwork are limited for commonsense reasoning results in Table 3. Do the authors have an explanation for this observation ? Are correlated weights less important for language tasks ?

### Soundness
2

### Presentation
3

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
This paper proposes DoRAN (Weight-Decomposed Low-Rank Adaptation with Noise Injection and Auxiliary Networks), a new variant of DoRA for parameter-efficient fine-tuning (PEFT) of large models. DoRAN aims to improve both stability and sample efficiency in low-rank adaptation. It introduces two main innovations: (1) a learnable noise injection term in the normalization denominator to stabilize gradients and preserve magnitude information adaptively; and (2) auxiliary hyper-networks that dynamically generate low-rank matrices across layers, enabling structural consistency and cross-layer parameter coupling. Theoretical analysis connects DoRAN to a Mixture-of-Experts framework and claims improved sample complexity from exponential to polynomial order. Experiments on both vision and language benchmarks show consistent improvements over LoRA and DoRA with minimal parameter overhead. Overall, the paper combines noise-based regularization and hypernetwork-driven generation into a unified PEFT framework targeting more stable and efficient fine-tuning.

### Strengths
- The authors provide a rigorous convergence analysis by reinterpreting DoRAN through the Mixture-of-Experts lens. Theoretical results establish improved sample complexity and offer insight into how shared parameterization and noise injection contribute to stability—an uncommon depth of analysis for PEFT papers.


- Experiments on diverse benchmarks (VTAB-1K, FGVC, LLaMA-7B/13B) demonstrate consistent improvements over prior PEFT baselines with minimal added parameters. Results support the claim that DoRAN improves both stability and sample efficiency, and the empirical section aligns well with the theoretical motivation.


- The paper is well-organized and clear in exposition. Mathematical derivations are transparent, figures are well-labeled, and the narrative effectively connects theoretical and empirical parts. Compared with prior DoRA-related works, this manuscript shows strong technical writing and coherent argumentation.

### Weaknesses
(1) **Lack of ablation on the noise injection mechanism.** The paper attributes DoRAN’s performance gain partly to the learnable noise term in the normalization denominator, but there is no ablation isolating its contribution. It remains unclear whether the improvement stems from the proposed stabilization or simply from regularization effects. A variant without the noise term or with fixed noise magnitude would clarify its real impact.

(2) **Limited analysis of the auxiliary hyper-network.** Although the auxiliary network is claimed to enable dynamic low-rank generation and cross-layer consistency, the paper does not show how these properties manifest in practice. No visualization or diagnostic (e.g., rank diversity, parameter correlation across layers) is provided. This makes it difficult to assess whether the hypernetwork truly improves representation capacity or merely adds parameter redundancy.

### Questions
- Could the authors provide an ablation study isolating the contribution of the noise term in the normalization denominator? For instance, comparing (1) no noise, (2) fixed Gaussian noise, and (3) learnable noise injection would clarify whether the improvement comes from true stabilization or from generic regularization effects. Including convergence or gradient variance plots would further strengthen this claim.

- It would be helpful to better understand how the auxiliary network influences cross-layer representations. Can the authors visualize or quantify how the generated low-rank matrices differ across layers, e.g., via rank spectra or inter-layer cosine similarit? Such analysis would make the "cross-layer consistency" argument more convincing and clarify whether the hyper-network provides structural benefits or just extra capacity.

### Soundness
2

### Presentation
3

### Contribution
2

# Mixture-of-Mamba: Enhancing Multi-Modal State-Space Models with Modality-Aware Sparsity

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
State Space Models (SSMs) have emerged as efficient alternatives to Transformers for sequential modeling, but their inability to leverage modality-specific features limits their performance in multi-modal pretraining. Here, we propose Mixture-of-Mamba, a novel SSM architecture that introduces modality-aware sparsity through modality-specific parameterization of the Mamba block. We evaluate Mixture-of-Mamba across three multi-modal pretraining settings: Transfusion (interleaved text and continuous image tokens with diffusion loss), Chameleon (interleaved text and discrete image tokens), and an extended three-modality framework incorporating speech. Mixture-of-Mamba consistently reaches the same loss values at earlier training steps with significantly reduced computational costs. In the Transfusion setting, Mixture-of-Mamba achieves equivalent image loss using only 34.76% of the training FLOPs at the 1.4B scale. In the Chameleon setting, Mixture-of-Mamba reaches similar image loss with just 42.50% of the FLOPs at the 1.4B scale, and similar text loss with just 65.40% of the FLOPs. In the three-modality setting, MoM matches speech loss at 24.80% of the FLOPs at the 1.4B scale. Our ablation study highlights the synergistic effects of decoupling projection components, where joint decoupling yields greater gains than individual modifications. These results establish modality-aware sparsity as a versatile and effective design principle, extending its impact from Transformers to SSMs and setting new benchmarks in multi-modal pretraining.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes Mixture-of-Mamba, a new SSM architecture for multi-modal learning that introduces modality-aware sparsity through modality-specific parameterization of the Mamba block. The approach extends ideas from Mixture-of-Transformers to SSMs and applies these techniques across settings involving text, image (both continuous and discrete token formats), and speech modalities.

### Strengths
1,Significant efficiency gains: mom achieves a good training loss while reducing flops.

2,Detailed ablation analysis: The authors analyze the impact of modality parameters on different projection matrices.

3,Simple and accessible architecture: The authors' pipeline is clear and easy to follow, providing an intuitive understanding of the structure.

4,Well-designed experiments: The authors compare against multiple strong baselines across multiple modalities, providing evidence of generalization.

### Weaknesses
1. All results focus solely on training and validation losses. The lack of direct evaluation on established downstream real-world tasks (e.g., image captioning, VQA, speech-to-text, and multimodal retrieval) limits its practical applicability.

2. More experimental details are needed, including GPU memory usage, parameter analysis, and training and inference time.

3. This approach employs rule-based modality-specific parameter selection, but is not directly compared to traditional MoE gating. No empirical verification or ablation tests are performed in the presented experiments to verify this claim.

4. Ablation experiments demonstrate the effect of decoupling the various projections but only highlight the joint effect. There is little discussion about why some projections (e.g., x_proj, dt_proj) individually provide small or negative gains — is this due to optimization, data-dependency issues, or overfitting? Some brief theoretical or empirical analysis could strengthen this argument.

5. More strong baseline comparisons are needed.

6. This work offers limited innovation. Its main innovation lies in adding modality-aware parameters to the projection layer of the SSM module. This type of work has been extensively explored in the Transformer field, combining knowledge from different domains with the original linear layer, such as MixLoRA, MoLE, and PHATGOOSE. The authors have simply replaced the knowledge from different domains with different modalities, which seems to fail to meet ICLR's requirements for high innovation.

7. Furthermore, the computational efficiency claimed by the authors appears to me to be achieved without modifying components that are not easily modified, such as A and selective_scan. The high efficiency comes from the original Mamba CUDA kernel, which is not the authors' contribution. Overall, I believe this paper's contribution is quite limited, and I do not recommend accepting it.

[1] https://arxiv.org/abs/2404.15159

[2] https://arxiv.org/abs/2404.13628

[3] https://arxiv.org/abs/2402.05859

### Questions
1. Downstream Evaluation: Can the authors provide results/evidence of the method's impact on downstream multi-modal tasks (such as VQA, image captioning, or speech-to-text)? Would these gains in loss and FLOPs translate to concrete performance or usability gains?

2. Variance Reporting: Can the authors include error bars, standard deviation, or results across multiple seeds for their reported training/validation losses to assess the stability and significance of reported gains?

3. Parameter Growth and Memory Cost: As the number of modalities grows, what is the impact on model parameter count and memory? Can the authors provide explicit numbers or analysis?

4. Learned Routing: Has the team implemented or compared to soft or learned gating/routing within the SSM framework? If so, how does that perform, and what are the failure/instability modes?

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
3

### Summary
This article primarily explores how to increase sparsity within the Mamba block in multimodal scenarios. The authors propose adding distinct proprietary parameters for different modalities within the Mamba block for processing, whereas many previous works have applied sparsification outside the Mamba block, such as in the MLP. Therefore, the approach proposed in this article is orthogonal to those previous works. Additionally, the authors conducted experiments under three different settings, and the results indicate that this model structure demonstrates improved efficiency in both image and text modality.

### Strengths
1. This article is well-written, making it relatively easy to follow
2. The solution proposed in this paper is orthogonal to the sparsification work previously done in Mamba, thus it has the potential to produce a cumulative effect.
3. The experiments are quite thorough, having been conducted under three different settings: multi-objective training, training with uniform representations, and training with three modalities. The results under these experimental setups all show good performance.

### Weaknesses
1. The article lacks some essential experimental settings. For example, although there are some structural changes to Mamba in this paper, some blocks are still shared. It is unclear whether the authors initialized these shared blocks using existing models. I couldn't find the relevant explanation in the article, and if I missed it, I would appreciate it if the authors could point it out to me.
2. In this article, the authors mainly use loss to monitor the model's performance. However, based on my experience, a lower loss does not necessarily imply better downstream performance in multimodal scenarios. Could the authors further explain why they did not use downstream task metrics?
3. This article lacks the necessary metrics for monitoring model performance. The author focuses most of the experiments on pre-training. Is this because the pre-trained model performs poorly in downstream tasks, such as multimodal understanding tasks, and therefore those metrics are missing? If so, I suggest that the author can test the model using few-shot learning methods or perform some fine-tuning and then conduct the testing.
4. For these modality-specific parameters, why not implement a dynamic selection mechanism that allows the model to automatically choose different parameters for different tokens?
5. Minor suggestion: some figures in the article are not very well formatted; for example, in Figure 3, some legends overlap with the curves.

### Questions
See weaknesses

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a Mixture-of-Mamba framework for multi-modal foundation model design. The core idea is to replace the projection layers in state-space model (SSM) modules with mixture-of-experts (MoE) modules. Overall, the framework is straightforward yet effective, demonstrating clear performance improvements across a range of model scales, datasets, and tasks.

### Strengths
- The proposed framework is conceptually simple and easy to integrate.
- Extensive experiments show consistent performance gains across multiple settings, underscoring the method’s general effectiveness.

### Weaknesses
- The main concern lies in the novelty of integrating MoE into the Mamba framework. Incorporating MoE into projection layers has been explored in numerous prior works, and the proposed adaptation appears incremental. The paper does not convincingly highlight what is unique about adopting MoE within Mamba.

- The paper lacks an in-depth analysis of how MoE facilitates multi-modal fusion in Mamba. Although experiments indicate that applying MoE to all projection layers yields the best performance, the underlying mechanism remains unclear. How does this compare to established cross-modal fusion strategies, such as Coupled Mamba or other variants?

- It is also difficult to understand why the MoE-based design outperforms dense Mamba. The paper should clarify what specific advantages are achieved by separating modalities within the projection layers and provide empirical or theoretical justification for these gains.

### Questions
Please see weakness

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper introduces Mixture-of-Mamba (MoM), a modality-aware extension of Mamba that applies structured sparsity inside the Mamba block rather than only in surrounding MLPs. Concretely, the core state-space projections are decoupled by modality so that only a subset is active per token/modality. Experiments span Transfusion, Chameleon, and a three-modality (text–image–speech) setup, with reports that MoM reaches comparable quality in ~2–2.5× fewer training steps at up to 1.5B parameters.

### Strengths
1. This method is effective for decoupling mamba blocks through different modality. 

2. Broad evaluation across multiple multi-modal pretraining regimes. Results suggest faster convergence in training steps versus dense Mamba and a flex-attention Transformer baseline.

### Weaknesses
1. Missing head-to-head comparisons with MoE-Mamba and BlackMamba under matched parameter and compute budgets. Without these, the claimed superiority or practical advantage is not fully substantiated.

2. Limited theoretical/diagnostic analysis explaining why performance depends on input modalities. A testable hypothesis is that specific subsets of parameters specialize to each modality; the paper does not analyze this.

3. The contribution is quite limited and does not meet the standard of publishing at ICLR, espcially considering the originality of the idea, which mainly extends Mixture-of-Transformers [1] to Mamba. 

[1] Liang, Weixin, et al. "Mixture-of-transformers: A sparse and scalable architecture for multi-modal foundation models." TMLR (2025).

### Questions
Regarding the modality-aware parameterization, are the three per-modality weight sets parallel (i.e., disjoint block partitions of the same projection), or is there a learned mask over a shared matrix that discovers the split?

### Soundness
2

### Presentation
2

### Contribution
1

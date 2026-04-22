# Revisiting Mixout: An Overlooked Path to Robust Finetuning

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 6, 4, 4, 4

## Abstract
Finetuning vision foundation models often improves in-domain accuracy but comes at the cost of robustness under distribution shift. We revisit Mixout, a stochastic regularizer that intermittently replaces fine-tuned weights with their pretrained reference, through the lens of a single-run, weight-sharing implicit ensemble. This perspective reveals three key levers that govern robustness: the \emph{masking anchor}, \emph{resampling frequency}, and \emph{mask sparsity}. Guided by this analysis, we introduce GMixout, which (i) replaces the fixed anchor with an exponential moving-average snapshot that adapts during training, and (ii) regulates masking period via an explicit resampling-frequency hyperparameter. Our sparse-kernel implementation updates only a small fraction of parameters with no inference-time overhead, enabling training on consumer-grade GPUs. Experiments on benchmarks covering covariate shift, corruption, and class imbalance, ImageNet / ImageNet-LT, DomainNet, iWildCam, and CIFAR100-C, GMixout consistently improves in-domain accuracy beyond zero-shot performance while surpassing both Model Soups and strong parameter-efficient fine-tuning baselines under distribution shift.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper revisits Mixout (Lee et al.) for robust fine-tuning of a pre-trained vision model under distribution shifts between train and test data. The authors motivate themselves by first reminding the implicit L2-penalty behavior of Mixout and then proposing a better alternative to Mixout, GMixout, by grounding it with an expected OOD loss decomposition proposed in the DiWA paper (Rame et al. 2022). By introducing an exponential moving average anchor with dynamic masking per episode, the proposed method enjoys a better tradeoff between ID and OOD performance compared to the considered baseline.


---

> Reference
- Lee at al. 2023, "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
- Rame et al. 2022, "Diverse Weight Averaging for Out-of-Distribution Generalization"

### Strengths
- Revisiting Mixout (Lee et al. 2023) in the robust fine-tuning setup offers interesting insights to the community.
- The proposed method is naturally connected to Mixout and DiWA (Rame et al. 2022), which have nice theoretical properties.
- The proposed method shows a performance gain across multiple distribution shift scenarios.
- The paper writing is clear and well-organized.

---

> Reference
- Lee et al. 2023, "Mixout: Effective Regularization to Finetune Large-scale Pretrained Language Models"
- Rame et al. 2022, "Diverse Weight Averaging for Out-of-Distribution Generalization"

### Weaknesses
- `Lack of technical/theoretical innovation`
  - The core technical contribution of the proposed method is the episodic update of the anchor (which is fixed as the pre-trained model weight in the original Mixout (Lee et al. 2023) and the Mask)
  - However, the idea of moving the anchor is already well explored by previous works on robust fine-tuning (Jang et al. 2024 -- periodic merging), and the merits of the exponential moving average (EMA) during fine-tuning are also well-explored by existing methods (Shu et al. 2023, Oh et al. 2024)
  - Although the authors bring a theory from DiWA paper (Rame et al. 2022) to explain the desired property of GMixout, there is **no rigorous analysis** (including supplementary A.1 that only provides a detailed decomposition) of **why GMixout achieves good control on the combination of variance-covariance-locality terms.**
- `Too limited baseline lineup and less comprehensive literature review`
  - In the experiment design, the authors **do not include some very representative robust fine-tuning methods**, such as WiSE-FT (Wortsman et al. 2022), LP-FT (Kumar et al. 2022), and FLYP (Goyal et al. 2022), which makes it difficult to gauge how significant the performance gain achieved by GMixout is.
  - Besides, they do not even mention some relevant works, CLIPood (Shu et al. 2023) and CaRot (Oh et al. 2024), where the **exponential moving average (EMA) style parameter update** was leveraged for the robust fine-tuning context, and VRF (Zhu et al. 2024), which **reduces variance for OOD robustness via ensemble between pre-trained and fine-tuned model prediction**, and DaWin (Oh et al. 2025), a state-of-the-art robust-fine-tuning method that is based on the **weight interpolation between pre-trained and fine-tuned models**.
    - Lack of citing these highly relevant works raises concern about the completeness of the authors' literature review.
- `(minor) Incorrect learnable parameter description`
  - I think the proposed method should have the same learnable parameters as Mixout (85.5 M) in Table 3.
  - Although the per-step updated parameter can be 9M (only survives after masking), as the authors sample the new mask for every episode, each episode has different updated parameters. 
  - Therefore, the learnable parameter over the whole training should be counted as 85.5 M (the same as the full FT).
  - And the GMixout is hard to recognize as a PEFT method, thereby.
---

> Reference
- Rame et al. 2022, "Diverse Weight Averaging for Out-of-Distribution Generalization"
- Jang et al. 2024, "Model Stock: All we need is just a few fine-tuned models"
- Wortsman et al. 2022, "Robust fine-tuning of zero-shot models"
- Kumar et al. 2022, "Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution"
- Goyal et al. 2022, "Finetune like you pretrain: Improved finetuning of zero-shot vision models"
- Shu et al. 2023, "CLIPood: Generalizing CLIP to Out-of-Distributions"
- Oh et al. 2024, "Towards Calibrated Robust Fine-Tuning of Vision-Language Models"
- Zhu et al. 2024, "Robust Fine-tuning of Zero-shot Models via Variance Reduction"
- Oh et al. 2025, "DaWin: Training-free Dynamic Weight Interpolation for Robust Adaptation"

### Questions
Please see the weakness section, and feel free to refute if there is any misunderstanding from me.

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
This paper studies and improves OOD robustness during fine-tuning by 1) revisiting MIXOUT, a stochastic regularization technique that replaces finetuned weights with pretrained references, and 2) proposes GMixout to improve robust finetuning of vision foundation models. The authors first analyze Mixout through bias-variance-covariance-locality (BVCL) decomposition to get an ensemble-based theoretical understanding. Then GMixout is designed with two key method-level modifications: (1) replacing the fixed pretrained anchor with an exponential moving average which can adapt during training and (2) controlling mask resampling frequency with a hyperparameter. Memory and compute efficiency is considered with sparse CUDA kernels. The method is evaluated across diverse distribution shift scenarios and demonstrates consistent improvements in OOD robustness with competitive ID accuracy.

### Strengths
- **Insightful initial theoretical analysis and motivation.** The paper studies a timely and important topic - OOD robustness of fine-tuning methods. The ensemble-based perspective on Mixout and the BVCL decomposition provides explanation for why mask sparsity, EMA coefficient, and resampling frequency matter for robustness. Then GMixout focuses on improving robustness with these three points.
- **Comprehensive experimental evaluation and ablation studies.** The paper evaluates on (1) a diverse set of benchmarks including different types of distribution shift (covariate shift, coruptions, long-tail) and (2) many baseline methods (full fine-tuning, PEFTs, and Model Soups), which strengthen the claims’ generalizability. Section 5.3 provides multiple ablation studies of different hyparameters, parameter budgets, and architectural choices (Vision vs. VL). GMixout shows higher OOD robustness while maintaining ID accuracy. 
- **Practical efficiency. The authors take memory and computational efficiency into consideration.** The sparse CUDA kernel implementation addresses a limitation of the original Mixout. This helps make the improved method feasible for finetuning large-scale models on consumer GPUs (table 3).

### Weaknesses
We thank the authors for submitting the paper to ICLR 2026! There are a few weaknesses listed below which I believe can make the paper better. For some points, please also refer to the questions below.
- **Missing statistical significance testing and inconsistent performance on large-scale data.** Throughout the paper, performance differences are often small (0.3-1 point) without error bars, confidence intervals, or significance tests reported. It makes it hard to tell whether improvements are meaningful or within noise margins. This is especially true for larger-scale data (as acknowledged in Observation 2 and table 4), where GMixout achieves only slightly higher average OOD robustness and lower robustness to some fine-tuning strategies (Model Soups and Random Mask). Also, more analysis and understanding on why GMixout sometimes underperforms would strengthen the contribution. 
- **Novel limitation.** GMixout is based on Mixout and the core modification (EMA anchor and resampling frequency) are relatively incremental. EMA-based weight averaging is well-established, and the resampling frequency is hyperparameter to tune. The connection to ensemble methods, which I find interesting to read, tends to be more interpretative than technically novel. But I think the idea of studying and combining all these techniques and the importance of this topic have a strong weight. I think this is not a big concern for me. 
- **Incomplete comparison with related work.** The paper does include a relatively thorough evaluation framework, but there are comparisons with a limited number of other recent robust fine-tuning methods beyond Model Soups and basic PEFT.

### Questions
- Can you provide error bars across multiple random seeds? Given the margins in many comparisons, this would strengthen the claims considerably. 
- Why does GMixout’s advantage narrow on ImageNet-1k compared to medium-scale datasets? Is this a fundamental limitation of the approach or coulter more hyperparameter tuning (episode I) help? 
- With long-tail dataset ImageNet-LT (table 2), GMixout achieves the best “few” shot accuracy and is behind Model Soups on “many” shot. Could you provide insight into this trade-off and whether it can be adjusted?

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
This paper addresses the trade-off between in-domain performance and robustness when finetuning vision foundation models. The authors revisit Mixout, interpreting it as an implicit ensemble via stochastic weight-sharing, and identify three key robustness factors: anchor choice, resampling frequency, and sparsity. They propose GMixout, which adapts the anchor using an exponential moving average and introduces an explicit resampling-frequency hyperparameter. A sparse-kernel implementation ensures efficiency with no inference overhead. Across benchmarks including ImageNet, DomainNet, iWildCam, and CIFAR100-C, GMixout improves finetuning accuracy while outperforming model soups and parameter-efficient baselines under distribution shift.

### Strengths
1. The paper is clearly written, well structured, and visually polished.
2. The experimental evaluation is extensive and convincingly supports the paper’s claims across diverse benchmarks.

### Weaknesses
1. The comparison is incomplete — several strong robust finetuning methods that dynamically constrain weight drift from pretrained models, such as TPGM[1], FTP[2], SPD[3], are not included.
2. The evaluation is limited to vision models; no experiments are provided on large language or vision-language models, which would strengthen the generality of the proposed approach.

[1] Tian, Junjiao, et al. "Trainable projected gradient method for robust fine-tuning." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[2] Tian, Junjiao, et al. "Fast trainable projection for robust fine-tuning." Advances in Neural Information Processing Systems 36 (2023): 11374-11393.

[3] Tian, Junjiao, Chengyue Huang, and Zsolt Kira. "Rethinking weight decay for robust fine-tuning of foundation models." Advances in Neural Information Processing Systems 37 (2024): 22418-22440.

### Questions
Please refer to the weaknesses.

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper revisits Mixout through the lens of implicit ensemble regularization, conceptually similar to stochastic regularization such as Dropout.
By doing this, this paper proposes GMixout, a generalized variant that improves supervised fine-tuning (SFT) OOD robustness under distribution shift.  
The authors argue that Mixout’s random parameter replacement implicitly forms an ensemble of subnetworks sharing the same backbone weights, and that controlling this stochastic process can enhance out-of-distribution (OOD) generalization.  
A sparse CUDA implementation further enables scaling to large ViT/CLIP backbones.
Reported image classification experiments across various benchmarks demonstrate consistent OOD robustness improvements compared to Mixout and other baseline methods, without sacrificing in-distribution (ID) accuracy.

### Strengths
This method empirically improves the OOD robustness.
The authors evaluate across multiple OOD settings and achieves superior OOD robustness in most settings.
he sparse CUDA implementation makes Mixout-style regularization feasible on modern large-scale vision models, demonstrating fair engineering contribution.

### Weaknesses
The method is effective but not very well motivated. Where does the improvement come from, why the adaptive anchor and the resampling frequence method is effective.  It will add to great value if more in-depth analysis could be made about the `implicit ensemble` feature of the GMixout process, instead of just describing them intuitively.

GMixout primarily modifies Mixout via some hyperparameters (EMA anchor and resampling frequency). While effective, it may be viewed as an engineering refinement rather than a conceptual breakthrough.

All reported experiments are conducted on vision classification tasks (ViT, CLIP).
However, the SFT-based algorithms, especially those PEFT methods, have also been primarily deployed to those more advanced models and/or tasks. For example, it is expected to report GMixout performance on LLM, VLM, or even those reasoning models, to justify its universal effectiveness and efficiency.

### Questions
Why model soup performs better on Cifar100?
The results in Table 4 seems not to be favor to the proposed method.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper revisits Mixout, a stochastic regularizer for finetuning pretrained models, and reinterprets it as a single run implicit ensemble mechanism. Based on this new perspective, the authors identify three core factors influencing robustness：1、Masking anchor：the reference weights that Mixout reverts to. 2、Resampling frequency：how often random masks are refreshed. 3、Mask sparsity：the proportion of weights replaced or retained. Building on this, the paper proposes GMixout, which (i) replaces the fixed pretrained anchor with an exponential moving average (EMA) of weights during training, and (ii) introduces a resampling frequency hyperparameter that controls how frequently subnetworks are resampled.The authors also present a sparse kernel GPU implementation that enables large scale finetuning on consumer GPUs without inference time cost. Extensive experiments on benchmarks such as ImageNet, DomainNet, iWildCam, CIFAR100 C, and ImageNet LT show that GMixout improves out of distribution (OOD) robustness while maintaining or improving in domain (ID) accuracy compared with LoRA, Random Mask, and Model Soups.

### Strengths
The work provides a novel theoretical reinterpretation of Mixout as an implicit ensemble in weight space；The proposed EMA based adaptive anchor and mask resampling frequency control are conceptually simple yet innovative extensions that directly improve robustness；The analysis using bias variance covariance–locality (BVCL) decomposition is insightful, linking ensemble theory to parameter efficient finetuning (PEFT).The experiments are comprehensive and rigorous, covering covariate shift, corruption, and class imbalance.Ablation studies (Figure 3，4) systematically examine how EMA coefficient, resampling frequency, and sparsity affect IDOOD trade offs.Results are consistent across multiple datasets and model sizes, showing strong empirical support.The insights may inspire further exploration of ensemble theoretic views of other finetuning methods (e.g., LoRA or adapters).

### Weaknesses
Limited exploration on language or multimodal tasks：Although the authors claim GMixout is general, all experiments are on vision datasets. Demonstrating its applicability on language or vision–language tasks would reinforce generality.

Comparison with newer PEFT baselines：The baselines include LoRA and Random Masking, but recent adapter free PEFT approaches (e.g., DoRA, AdaLoRA, and QLoRA) are not discussed empirically. Including would provide stronger positioning. Although the ensemble based bias，variance，covariance，locality analysis is conceptually appealing, the theoretical derivation is mostly heuristic. A more formal proof or tighter bounds on the expected OOD error under Mixout/GMixout would strengthen the argument.

### Questions
1. refer to weakness

2. Can GMixout be applied to text based or multimodal foundation models (e.g., CLIP, LLaVA, or language- only transformers)? If so, are there any expected differences in behavior? and how sensitive is performance to λ and k when scaling to larger models like ViT-L/14 or other architectures (e.g., ResNet backbones)?

### Soundness
3

### Presentation
2

### Contribution
2

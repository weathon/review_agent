# SeeDNorm: Self-Rescaled Dynamic Normalization

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6

## Abstract
Normalization layer constitutes an essential component in neural networks. In transformers, the predominantly used RMSNorm constrains vectors to a unit hypersphere, followed by dimension-wise rescaling through a learnable scaling coefficient $\gamma$ to maintain the representational capacity of the model. However, RMSNorm discards the input norm information in forward pass and a static scaling factor $\gamma$ may be insufficient to accommodate the wide variability of input data and distributional shifts, thereby limiting further performance improvements, particularly in zero-shot scenarios that large language models routinely encounter. To address this limitation, we propose SeeDNorm, which enhances the representational capability of the model by dynamically adjusting the scaling coefficient based on the current input, thereby preserving the input norm information and enabling data-dependent, self-rescaled dynamic normalization. During backpropagation, SeeDNorm retains the ability of RMSNorm to dynamically adjust gradient according to the input norm. We provide a detailed analysis of the training optimization for SeedNorm and proposed corresponding solutions to address potential instability issues that may arise when applying SeeDNorm. We validate the effectiveness of SeeDNorm across models of varying sizes in large language model pre-training as well as supervised and unsupervised computer vision tasks. By introducing a minimal number of parameters and with negligible impact on model efficiency, SeeDNorm achieves consistently superior performance compared to previously commonly used normalization layers such as RMSNorm and LayerNorm, as well as element-wise activation alternatives to normalization layers like DyT.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new normalization method called SeeDNorm.
The purpose of this normalization method is to address the wide-ranging variations and distribution shifts in input data that a static scaling factor γ cannot handle.
The proposed method is simple yet effective.
Its effectiveness is discussed in the context of large-scale language model pre-training, as well as supervised and unsupervised vision tasks (such as image generation).

### Strengths
- The paper is well-structured and easy to understand.
  - It strikes a good balance between theoretical explanation and experimental validation.
- Experiments are conducted on both language models and vision tasks, discussing effectiveness across a broad range of domains.
- References are appropriately cited.

### Weaknesses
- I consider this a well-written paper. None of these points represent fatal flaws as a paper; they are comments to better understand its potential.

- While evaluating results on OLMo2 as a language model, demonstrating effectiveness on other language models would further strengthen the paper's persuasiveness.
  
- Regarding vision models, effectiveness has been confirmed for CNN-based models like ConvNext. I consider this a very positive point. However, for CNN-based models, techniques beyond Batch Norm exist, such as GroupNorm, Weight Normalization, and Weight Standardization. Is comparison or combined use with these possible?

### Questions
- I fundamentally believe this is a well-written paper. These are questions to better understand the paper's potential. Please see the weaknesses section for details. 
  - Is the proposed method effective with language models other than OLMo2?
  - Regarding the possibility of combining it with other normalization techniques, if you already have insights, please share them.
  - Is the implementation planned to be released?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper introduces SeeDNorm, a novel normalization layer for neural networks. SeedNorm is an extension of RMSNorm, that modulates the scale parameter gamma of RMSNorm by adding a term to gamma which consists of two learnable parameters. These parameters act on the unnormalized input of the normalization layer, and thus allow scale information to flow through the normalization layer. The authors prodide a gradient analysis of SeeDNorm in order to show that their normalization layer is insensitive to extremely large or small values, and to justify their regularization and initialization choices. They provide experiments with OLMo models for LLM pretraining, DiTs for Image Generation, ViTs and ConvNext for Image Classification, and ViTs for Self-supervised learning, overall reporting improvements over baselines like LayerNorm and DyT. They also provide ablations on their design choices.

### Strengths
- I appreciate the large scope of the experiments: The paper presents extensive studies across tasks and scale, spanning LLM pretraining (with normal and MoE models), image classification, and image generation. 
- SeeDNorm either matches or outperforms other normalization variants. The strongest benefits are observed for MoE language models 
- The ablations are interesting and comprehensive. They answer several questions that came up when reading through the paper
- the paper is well-written and easy to follow

### Weaknesses
- A few experimental details are missing, and it is unclear if default training scripts or modified scripts have been used (more below in questions)
- For Vision experiments, especially image classification, the benefits of SeeDNorm are less clear. In Table 2, the numbers are either identical or extremely close to baselines (yet the SeedNorm variant is bold - why?). Further, the default SeedNorm variant fails to converge (Table 3), and Multihead SeedNorm is required to match or bring gains over LayerNorm or DyT, with different choices of head dimension often leading to suboptimal performance. Also the drop path rate had to be adjusted for large models “to further prevent overfitting”. This questions if SeeDNorm can be used as an easy drop-in replacement across tasks
- as for the vision experiments, the numbers for OLMo are sometimes close. Since there are no error bars or runs over multiple seeds (which is partially understandable, given the computational cost of large scale experiments), it is hard to judge the significance of the results
- In Theorem 3.2 the authors claim that the dot product of two random vectors is inversely proportional to their dimension D. However, they then proceed to show that it is proportional (not inversely) to D. If it were indeed inversely proportional, splitting the vectors to reduce the variance would not make sense

Minor remarks:  
- typo in line 28, neglligible
- line 1110: “Our method similarly achieved consistent improvements” ( OLMo2-1B), but this varies across tasks, and on average, the accuracy is almost identical
- in the abstract in line 18, the connection between a static scaling factor zero-shot setups (as compared to other setups) is not clear to me

### Questions
- How was the training setup for the LLM experiments chosen? Is it the default OLMo recipe? 
- in line 302 the authors write that the parameter $\alpha$ is initialized to 1 for language experiments. I could not find if it is initialized differently for vision experiments. Can the authors clarify this?
- How was the weight decay for $\alpha,\beta$ chosen? Was it the default weight decay used for other parameters, or was it tuned?
- the numbers in Table 2 are exactly identical to the numbers in the DyT paper. Is this because the authors used the same codebase? Or were the numbers taken from that paper?
- Overall, was there any form of hyperparameter tuning or adaption involved for SeedNorm, or were always the default recipes taken from somewhere (where?), and only the normalization layer was replaced?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes SeeDNorm (Self-Rescaled Dynamic Normalization), a dynamic normalization method that preserves input norm information and adaptively adjusts the scaling factor based on the current input. The work aims to address the limitation of RMSNorm, which discards the input norm during the forward pass and relies on a fixed scaling factor that cannot adapt to data variability or distributional shifts, leading to suboptimal performance in zero-shot scenarios. SeeDNorm introduces a self-rescaling mechanism that modulates the scaling coefficient according to the input itself, thereby retaining norm information and enabling data-dependent normalization. The paper provides theoretical analysis of scaling invariance and gradient behavior in both forward and backward propagation, showing that SeeDNorm maintains numerical stability and adaptively adjusts gradients according to input norm. Experiments demonstrate consistent improvements over RMSNorm, LayerNorm, and DyT across both language modeling (dense and MoE LLMs) and computer vision (ViT, ConvNeXt, DiT, MAE) tasks, with faster convergence and better zero-shot performance achieved with negligible additional parameters and minimal computational overhead.

### Strengths
The paper is technically detailed and the motivation is clear: existing normalization layers such as RMSNorm provide stability but lose information about the input magnitude. SeeDNorm offers a straightforward extension that dynamically rescales activations conditioned on the current input, improving adaptability to data variability and distributional shifts. The theoretical analysis is comprehensive, covering forward and backward derivations as well as scaling invariance, and helps clarify why the method remains stable. The experimental section is broad and systematic, spanning both language and vision tasks, and the results consistently show better convergence and downstream performance. Ablation studies on activation choice, initialization, weight decay, and the multi-head formulation strengthen the empirical support. The method is simple to integrate and performs reliably across different architectures.

### Weaknesses
Although the results are strong, the contribution may appear incremental because SeeDNorm can be viewed as a combination of RMSNorm and DyT that merges their strengths while mitigating DyT’s vanishing gradient issue. The proposed dynamic rescaling mechanism resembles existing modulation approaches such as gating or FiLM-style conditioning, differing mainly in how the rescaling term is computed. The theoretical analysis mainly focuses on stability and gradient behavior but gives limited insight into why the dynamic adaptation improves representation learning or generalization. The paper states that the computational overhead is negligible, but there is no quantitative evidence such as runtime or FLOPs comparison. In addition, the multi-head version is only applied to vision tasks, and it is unclear whether a similar strategy could benefit language models or whether it was found unnecessary.

### Questions
- How do $\alpha$, $\beta$, and $\gamma$ evolve during training? Do they converge to stable values or continue adapting dynamically?
- In Figure 3 (fourth subplot), SeeDNorm shows a temporary spike in loss; is this caused by instability in the rescaling term or by random training noise?
- Why is the multi-head variant used only for vision models? Would multi-head rescaling help stabilize training in language models, or is it redundant in that setting?
- Can the authors provide quantitative measures of computational overhead, such as runtime or FLOPs, to support the efficiency claim?
- How does SeeDNorm interact with normalization-free or re-scaling transformer variants, and could it complement them?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The authors point out that existing normalization layers, such as RMSNorm and LayerNorm, overlook the norm and scale information of the input. In situations where there are distributional shifts, the norms of the input can provide valuable information and a consistent scaling factor. The authors argue that ignoring this aspect reduces the expressiveness of models, particularly in large language models and vision models that operate across different domains. To address this issue, the proposed SeeDNorm introduces a dynamic scaling factor for normalization that is based on the input itself. This approach helps to retain some norm and scale information while adapting to each individual input sample. When training large-scale transformers, whether for language or vision tasks, and when facing varying input distributions, it may be beneficial to consider this proposed method as it could potentially enhance performance by avoiding the limitations of fixed normalization.

### Strengths
1. SeeDNorm offers improved preservation of input norm information and adaptability to shifts in input distribution. Unlike RMSNorm, which disregards input norms, SeeDNorm retains some of the "scale" information through a dynamic term. This scaling, being dependent on the input, allows it to better manage varying magnitudes or changes in the domain.

2. Experiments conducted across language tasks demonstrate that SeeDNorm accelerates convergence and produces better final results compared to the baseline methods of RMSNorm and LayerNorm.

3. The paper not only presents empirical results but also includes the derivation of gradients, discusses invariance properties, and addresses initialization and regularization choices for stability.

### Weaknesses
1. For extremely large models or those with many layers of normalization, the benefits of SeeDNorm may be limited. The paper indicates that the improvements diminish for dense models compared to Mixture of Experts (MoE) models.

2. Although SeeDNorm shows consistent improvement, the gains are often subtle, typically just a few tenths of a percent in accuracy, particularly for dense models, as illustrated in Table 2.

3. While the paper broadly discusses "vision tasks," it does not assess tasks that heavily involve spatial features, such as detection, oriented bounding boxes (OBB), and segmentation. More targeted experiments would strengthen the validity of their claims.

### Questions
1. How beneficial would SeeDNorm be in smaller models or resource-constrained environments, such as mobile or edge devices where memory and speed are crucial? Could the authors provide some insights on this?  
2. How stable and easy is it to integrate SeeDNorm into different network architectures?  
3. Does SeeDNorm generalize across various tasks without requiring extensive tuning?

### Soundness
3

### Presentation
3

### Contribution
2

# Boomerang Distillation Enables Zero-Shot Model Size Interpolation

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 4

## Abstract
Large language models (LLMs) are typically deployed under diverse memory and compute constraints. Existing approaches build model families by training each size independently, which is prohibitively expensive and provides only coarse-grained size options. In this work, we identify a novel phenomenon that we call boomerang distillation: starting from a large base model (the teacher), one first distills down to a small student and then progressively reconstructs intermediate-sized models by re-incorporating blocks of teacher layers into the student without any additional training. This process produces zero-shot interpolated models of many intermediate sizes whose performance scales smoothly between the student and teacher, often matching or surpassing pretrained or distilled models of the same size. We further analyze when this type of interpolation succeeds, showing that alignment between teacher and student through pruning and distillation is essential. Boomerang distillation thus provides a simple and efficient way to generate fine-grained model families, dramatically reducing training cost while enabling flexible adaptation across deployment environments. The code and models are available at [https://github.com/dcml-lab/boomerang-distillation](https://github.com/dcml-lab/boomerang-distillation).

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces and empirically validates a novel phenomenon termed "boomerang distillation." The method provides a highly efficient, zero-shot technique for creating a family of models with fine-grained sizes that interpolate between a small, distilled "student" model and a large "teacher" model. The authors conduct extensive experiments across multiple model families (Qwen, Pythia, Llama) and sizes, demonstrating that this method produces models whose performance scales smoothly with size. These interpolated models consistently and significantly outperform naive layer pruning and often match or even surpass the performance of models of equivalent size that are trained individually via standard distillation, all while dramatically reducing the computational cost of creating model families.

### Strengths
- This article proposes a simple, three-stage scheme for zero-shot model size interpolation, which is both novel and highly practical. The ability to generate a series of model sizes after just one training run significantly reduces the computational resources required to build and deploy model families.
- The paper’s claims are supported by extensive and well-designed experiments. This method has been validated across multiple architectures (Qwen, Pythia, Llama) and scales, demonstrating its universality. The comparison with relevant baselines is thorough and effectively highlights the superiority of the proposed technique.
- The paper is exceptionally well-written and presented. Figure 1 provides a clear and intuitive overview of the entire process. Figures 2, 3, and 4 are particularly effective in visualizing core results: the smooth performance interpolation of boomerang distillation contrasts with the poor or unstable performance of the baselines. These figures make the core claims convincing and easy to grasp.
- The ablation studies are also highly comprehensive, providing crucial support for the paper’s conclusions.

### Weaknesses
- The article mentions an inconsistency in performance between Llama-3.2-3B and the Qwen/Pythia Family. This limits the universality of the Boomerang Distillation method, suggesting that its successful application may require non-trivial, model-specific analysis—slightly undermining the "zero-shot" simplicity of the patching step.
- The article notes in Appendix E.1 that there is a lower bound for model compression when using this method. However, consider a scenario where we aim to distill DeepSeek-R1, a 671B model: in practice, we can obtain a 1.5B model through data distillation. This size reduction far exceeds the experimental lower bound of the method, revealing certain limitations of Boomerang Distillation.

### Questions
- Would the same effect still hold if simple modifications are made to the student model? For instance, the internal structure of the student model might differ from that of the teacher model—such as altered activation functions—even when their hidden dimensions are identical. If the current effect remains unchanged, what conclusions could be drawn from this?
- In Section 3.2 of the article, it is mentioned that low-quality datasets lead to the need for direct model distillation methods like Boomerang Distillation. Could openly available high-quality datasets be used to validate this conclusion?

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper is concerned with knowledge distillation for dynamic inference. It observes that by treating the layer-dropped LM as the student LM and the original LM as the teacher LM, the student LM could dynamically adjust the inference budget by inserting dropped layers again into the student LM. With the inserted layers, the performance of the student LM could also be improved. The method proposed is termed boomerang distillation in this paper. The method enables a fine-grained model family with only one pass of distillation, reducing the expected cost.

### Strengths
1. It is actually a very counter-intuitive observation that the layer-dropped student could be recovered by adding the dropped layers back.
2. The one-pass distillation essentially enables efficient model family derivation, which may serve as a cost-effective way in both academia and industry.

### Weaknesses
1. The experiments are rather limited to model scales smaller than 10B, which might constrain the application of the method to larger-scale LMs.
2. In Figure 2, there seems to be a sudden performance improvement in generation accuracy along the insertion of layers, however, the improvement in classification accuracy is rather smooth. Further elaborations on the phenomenon should be attached here.

### Questions
N/A

### Soundness
3

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
5

### Summary
This work explores an interesting phenomenon termed Boomerang Distillation. The authors discover that, under certain constraints, adding back layers from a distilled model can produce a spectrum of intermediate model sizes without any additional training. They further show that these interpolated models achieve performance comparable to standard distilled models of the same size.

### Strengths
- The phenomenon presented is both interesting and novel.
- The presentation is clear, and the experimental results are comprehensive and well-executed.

### Weaknesses
- The paper lacks an in-depth explanation of the phenomenon. While a rigorous theoretical analysis may be unrealistic, including some qualitative observations or a proof-of-concept experiment to illustrate the underlying rationale would be beneficial.
- In the related work section, the authors compare their approach to model interpolation and claim that prior interpolation methods operate solely in weight space. However, interpolating across different architectures can also be regarded as a special case of weight-space interpolation (where certain weight matrices are zero). Moreover, LlamaFlex[1] discusses interpolation between models of different sizes in its “Router Interpolation” section, which seems relevant and worth acknowledging.
- This work focuses exclusively on constructing student models through layer dropping. It would be valuable to discuss whether the observed phenomenon also applies to width or head pruning. While interpolation in these cases may be more challenging, such discussion would provide a more complete perspective.

[1] LLaMaFlex: Many-in-one LLMs via Generalized Pruning and Weight Sharing

### Questions
None

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces Boomerang Distillation, a novel approach for knowledge distillation that allows zero-shot model size interpolation. The key idea is to train the teacher and student jointly in a bidirectional distillation loop (“boomerang” scheme), where gradients from both directions are exchanged in a dynamically balanced manner. The method achieves smooth performance interpolation across different model scales and architectures (e.g., ViT and ResNet families), showing competitive results on ImageNet, CIFAR, and transfer benchmarks, while maintaining efficiency and data-free adaptability.
The paper introduces a technique termed boomerang distillation, in which a large “teacher” model is first distilled into a smaller “student” model, and then a family of intermediate-sized models is generated without further training by gradually re-inserting blocks from the teacher into the student. The resulting models interpolate in size (number of parameters) and in performance smoothly between student and teacher, and in many cases match or outperform traditional distilled models of the same size. The authors analyse the necessary conditions and compare the method to layer-pruning baselines, showing favorable results.

### Strengths
1.Enables flexible interpolation between different model sizes without retraining, which is practical for model deployment across devices.
2.Demonstrated across multiple architectures and datasets with consistent improvements and smooth interpolation curves.
3.The paper goes into details about what components are necessary: e.g., the alignment loss, the initialization from teacher weights, etc. They also compare against layer-pruning baselines. This helps the community understand when/why the method works.

### Weaknesses
1.It remains unclear how the method scales to much larger teacher models (tens or hundreds of billions of parameters) and whether the interpolation remains as clean in those regimes.
2.The paper focuses on language models. It would strengthen the work to show applicability in other modalities (vision, multimodal) or architectures for broader impact.
3.While no further training is required for each intermediate size, the approach still requires the initial distillation and alignment of a student model—which may itself be costly.
4.The method assumes that student and teacher share compatible hidden dimensions and layer structure to allow patching. This may limit generality to heterogeneous architectures or differently sized hidden layers.

### Questions
1.How sensitive is the method to the relative capacity of teacher and student (e.g., extreme size gaps)?
2.If the teacher and student have different hidden dimensions, can patching still be applied?
3.If the model is multimodal with fusion layers, could inserting teacher blocks disrupt cross-modal interactions, and how might you control or evaluate that risk?
4.What is the practical GPU cost increase compared to standard KD?

### Soundness
3

### Presentation
3

### Contribution
3

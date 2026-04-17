# QPrompt-R1: Real-Time Reasoning for Domain-Generalized Semantic Segmentation via Group-Relative Query Alignment

- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Deploying semantic segmentation in driving and robotics requires both real-time inference and robustness to domain shifts, formalized as Real-Time Domain-Generalized Semantic Segmentation (RT-DGSS), which has not been fully addressed. Existing methods often treat real-time(RT) inference and domain generalization (DG) separately, with DG improving robustness but lacking real-time performance, and real-time models being brittle under distribution shifts. To address the RT-DGSS problem, we propose QPrompt-R1, a real-time Query-Prompt architecture built on a ViT backbone. QPrompt-R1 injects a small set of learnable queries only at the final transformer block, performing a single query–image alignment step and eliminating decoder overhead. To further enhance alignment without test-time cost, we introduce a Group Relative Query Alignment (GRQA) objective, which uses group-relative supervision within each group to align queries with features, improving domain generalization through group-relative rewards. QPrompt-R1 achieves 54 FPS, delivering strong performance in synthetic-to-real transfer, real-to-real generalization, and robustness under adverse conditions. Additionally, GRQA is plug-and-play, improving state-of-the-art DGSS methods like REIN (+1.2) and SoMA (+0.5) without inference-time overhead.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposed a unified framework to address real-time segmentation and domain generalized segmentation, which is relevant to real-world applications. The proposed is a plug-and-play solution, which can benefit existing methods without introducing latency overhead. Experiments and ablation studies demonstrate the effectiveness of the proposed method.

### Strengths
- This paper provided insights to address real-time segmentation and domain-generalized segmentation in a single framework.
- The proposed GRQA is a plug-and-play module, which can enhance existing real-time segmentation models.

### Weaknesses
- As the proposed GRQA is a plug-and-play module, it is also necessary to evaluate atop high-quality, non-real-time segmentation models to verify whether it is consistently beneficial.
- The related work only reviewed CNN-based real-time segmentation methods. More recent state-of-the-art real-time methods like SeaFormer, SeaFormer++ RTFormer, and Next-ViT should be reviewed and compared in the experiments. Moreover, the domain generalization performance under different corruptions should be evaluated on Cityscapes-C and relevant benchmarks and compared against existing methods like SegFormer, FAN, Robustifying Token Attention for Vision Transformers, and Trans4Trans.

### Questions
- Would you consider making the source code publicly available to foster future research in this line?
- As this work claimed to address real-time efficiency and domain generalization in a single framework, would you better justify the benefit of your solution for reducing the parameters of the model and improving the inference speed? This is critical and should be discussed in detail.

### Soundness
2

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
3

### Summary
This work proposes QPrompt-R1, a method specifically designed for RT-DGSS (real-time domain generalized semantic segmentation). QPrompt-R1 consists of two components: the QPrompt architecture that ensures real-time performance and GRQA that enhances domain generalization. Based on the VFM backbone, QPrompt injects learnable queries into the last Transformer block to replace the traditional complex decoder. GRQA addresses the insufficient robustness of single-layer query-image interaction. Experiments demonstrate notable performance improvement.

### Strengths
- It effectively applies GPRO to segmentation tasks, enhancing segmentation performance without introducing additional inference overhead.
- GRQA can be effectively integrated into current domain generalization semantic segmentation methods to further improve their performance.
- QPrompt-R1 strikes a good balance between domain generalization capability and inference speed.

### Weaknesses
- Q-Prompt is similar to EoMT. They both insert queries into the final layer, but the authors discuss no differences.
- In the Evaluation Setting of Section 4.1, the Real→Real scenario should exclude ACDC’s adverse-condition splits.

### Questions
- I am curious about whether GRQA is capable of improving the performance of general semantic segmentation. Discussion or even a small experimental analysis would furthur improve this work.
- As the proposed method uses prototye and prototye bank, it is suggested to cite CPT [r1] where category prototype and memory bank are also adopted.

[r1] Quan Tang, et al. Rethinking Feature Reconstruction via Category Prototype in Semantic Segmentation,TIP 2025.

I am more familiar with general semantic segmentation techniques, and follow only mainstream ones in domain transfer. Feel free to correct if there is any misunderstanding.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper investigates the problem of real-time domain-generalizing semantic segmentation. The authors propose to depart from standard, heavy decoders which are used for this dense classification task on top of vision foundation model encoders, and replace these decoders with a single, light-weight and efficient transformer decoder layer. To enrich the mask-level queries which are involved in this decoder layer, they propose a novel group-relative query alignment module which produces a reinforcement-learning-type reward, optimized in a standard supervised fashion. The authors validate their model on several central domain generalization benchmarks, achieving very high efficiency while staying competitive in terms of segmentation performance to state-of-the-art generalizable segmentation methods.

### Strengths
1. The examined problem, albeit constituting a simple combination of two existing ones - i.e. real-time semantic segmentation and domain-generalizing semantic segmentation, is underexplored and yet very relevant for practical real-world applications of Computer Vision, including safety-critical settings that involve uncontrolled and unconstrained operational design domains and at the same time fast inference, such as autonomous vehicles and field robotics.

2. The idea to depart from complex, MaskFormer-style decoders which are the current de-facto standard for semantic segmentation and to leverage the features which are already extracted by Vision Foundation Models in their image tokens is interesting and promising, even though the authors have not provided experimental evidence that the bottleneck of recent segmentation models in terms of latency are indeed these decoders.

3. The proposed Group-Relative Query Alignment is a novel technique in the context of semantic segmentation, with a strong and rigorous theoretical formulation and a demonstrated clear performance benefit for diverse VFM-based segmentation networks (cf. Table 6). This novel module is relevant both in the context of real-time semantic segmentation and beyond that, as it delivers performance gains for heavier, more highly performant yet less efficient networks for domain-generalizing semantic segmentation (cf. Table 2).

### Weaknesses
1. Missing evidence for motivation on efficiency-level design decisions. The authors state in the very beginning of the paper in the Abstract that "the bottleneck in Domain Generalization is the prediction head, not the backbone". However, this statement is not supported by experimental evidence, e.g. by benchmarking a baseline network of similar architecture to the one proposed which however does include standard pixel and transformer decoders - a la Mask(2)Former - in terms of inference computation in order to break down the inference latency / FLOPS into backbone-level and head-/decoder-level portions. Combined with the fact that in the main experimental comparison (Table 1) the proposed model trails the performance of state-of-the-art, heavier methods, most of which feature the above decoders, the precise quality of the tradeoff between performance and efficiency is not clear.

2. Omission of learned upsampling to handle fine-grained details in segmentations. The utilized VFMs, such as ViT, rely on patchification to tokenize the input image, which heavily reduces the resolution of their output image-token-based features. Yet, the proposed method operates directly on these heavily downsampled features to produce the final segmentation map. That is, the method does not involve any learned upsampling, exactly of the type that is performed in the pixel decoder module of standard mask-based segmentation architectures such as Mask(2)Former which is omitted in the presented architecture. Thus, the question arises on whether the resulting predictions are deteriorated in terms of preservation of fine-grained details in the ground-truth semantic boundaries.

3. Issues with the presentation of the method. The overall presentation of the method is of fair quality, however, there are a few unclear points. The symbol $K$ is never properly defined and it is not clear whether it denotes the number of semantic classes in the set, as is usually the case in the notation of semantic segmentation papers. This potential relation is reinforced by the fact that in Fig. 4, the number of queries $K$ is 20, which differs from the number of class prototypes (19) only by 1. In Eq. (7) and (8), the parameter $\epsilon$ appears twice; only a single addition of a positive constant is nonetheless required to avoid an ill-defined zero-valued denominator in (8). The core motivation behind the full GRQA objective introduced in p. 6 is not straightforward to me. Some background on this type of reinforcement learning could be useful for readers who are not familiar with the topic. In this context, L. 268-271 are unstructured and hard to parse.

### Questions
1. Can the authors provide an "end-to-end" ablation starting from a baseline, heavy network with pixel and transformer decoder heads on top of the VFM backbone, and then first remove these heads and subsequently add the proposed components one by one, finally arriving at the fully-fledged proposed network? Such an ablation should include both performance figures and efficiency figures, importantly latency and amount of computation, and potentially also parameter count and total space complexity at inference. This would greatly help in supporting the motivation that efficiency can be substantially improved by removing the decoder heads.

2. Can the authors provide clarifications on the GRQA part of their method? (cf. Weaknesses)

3. Can the authors provide a comparison in terms of boundary accuracy of the competing segmentation methods against their method? This would help to understand whether the proposed efficiency improvement preserves fine-grained details in the segmentation output despite working with lower-resolution features.

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
3

### Summary
The paper proposes QPrompt-R1, a method addressing the novel and practical problem of Real-Time Domain-Generalized Semantic Segmentation (RT-DGSS). The core idea is to inject a small set of learnable queries into the final layer of a Vision Foundation Model (VFM) backbone (QPrompt) and enhance their training with a novel Group-Relative Query Alignment (GRQA) objective. This approach aims to retain the domain generalization benefits of query-based methods while achieving real-time speeds by avoiding heavy decoder stacks. The results are impressive, showing a strong balance of speed and robust cross-domain performance.

### Strengths
1. Generalizability of GRQA: The demonstration that GRQA is a "plug-and-play" module that can be added to other SOTA DGSS methods to improve their performance is a major strength.
2. Motivation: The paper is well-motivated, identifying a critical and practical problem.

### Weaknesses
1. Overclaiming of the "First" RT-DGSS Problem: While this paper is the first to formally name and define the problem, the concept of achieving real-time and robust segmentation is an implicit goal in the field. RTSS papers also evaluate cross-domain performance and DG papers evaluate the computational efficiency, even if it's not their primary focus.
2. Novelty of QPrompt: The idea of injecting tokens/queries is not new. The prototype bank, which is claimed to be a key component of the GRQA strategy, has already been widely applied. The specific application to create a lightweight, single-layer "prompting" mechanism for segmentation might be novel in this context, but the framing should be more precise to acknowledge the inspiration.

### Questions
Baseline Comparison: Could you include a comparison with a prompted version of SAM or other fast, promptable foundation models to better contextualize the performance of your QPrompt architecture?

### Soundness
3

### Presentation
2

### Contribution
3

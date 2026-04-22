# Learning frequency domain codes for semantic vision

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 8, 2, 4

## Abstract
Visually semantic concepts such as objects and categories provide a natural foundation for semantic reasoning, yet standard deep learning-based vision models routinely extract and aggregate features using homogeneous stacks of spatial layers. As a result, feature representations are learnt implicitly without clear organisation, rendering decision-making processes opaque and difficult to interpret. Psychovisual processing provides a way to mimic how the brain encodes and interprets visual information that produces higher abstractions from low-level processing. In this paper, we propose Semantic Visual Coding (SVC), a learnt frequency domain representation that introduces explicit psychovisual abstraction into convolutional neural networks (CNNs).  Inspired by psychovisually motivated image codes from the 1990s, SVC learns band-limited filters that encode task-relevant semantics as distinct regions of the frequency domain. These converge towards sparse (data-driven) coronal patterns that suggest a natural representation scheme for semantic abstractions supporting model reasoning. We also introduce a framework that adapts CNNs to be psychovisually aware by combining traditional low-level spatial feature extraction with high-level abstraction in the frequency domain via SVC, which we call 'PsychoNet'. Salience analyses show that PsychoNet’s spatial layers extract highly interpretable object parts and morphological features, unlike blob-like regions produced by standard CNN. It further finds that SVC forms structured selections of these parts that are organised by spatial scale, suggesting frequency domain abstraction as a promising direction for interpretable models which reveal the semantic features they employ.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces Semantic Visual Coding (SVC), a learned frequency domain representation for encoding high-level visual features in convolutional neural networks. The authors develop PsychoNet, an architectural framework that adapts ResNet and ConvNeXt models to operate in both spatial and frequency domains, inspired by psychovisual processing concepts from Saadane et al. (1998). PsychoNet employs spatial layers for low-level feature extraction and frequency domain processing via SVC for high-level abstraction and reasoning. The framework is evaluated on CIFAR-10, CIFAR-100, ImageNet-100, and ImageNet-1K classification tasks, where it achieves comparable or slightly improved performance relative to baseline ResNet models, though it underperforms slightly against ConvNeXt-S on ImageNet benchmarks. Analysis of learned representations reveals that SVC converges to sparse, data-driven frequency patterns, while spatial layers extract interpretable object parts.

### Strengths
**Novel architectural contributions:** The paper introduces a dual-domain processing framework that separates low-level image features extraction from high-level frequency domain abstraction through what they name the PsychoNet architecture.

**Comprehensive background and contextualization:** The introduction provides thorough coverage of related work, effectively positioning the contributions within the existing literature.

**Detailed architectural exposition:** Substantial space is dedicated to explaining the components of PsychoNet.

**Reproducibility commitment:** The authors commit to releasing code including training scripts, model weights, and instructions.

### Weaknesses
**Unclear practical motivation and overstated contributions:** The domain or application that would benefit from this work remains unclear throughout the paper. The introduction overstates contributions by claiming:
- PsychoNet "maintains or improves the performance of common and state-of-the-art CNNs" (line 86), when performance improvements are limited to ResNet baselines and the model performs on par with ConvNeXt-S.
- "SVC performs abstraction and reasoning in the frequency domain" (line 88) contradicting the limitations section acknowledgment that "it is not clear how these representations are used for reasoning, which remains an important direction for future work" (line 430).

**Weak foundational work:** Saadane et al. (1998), the foundational psychovisual work upon which this paper builds, has only 8 citations, with 5 from the same authors. This raises questions about the influence and validation of the underlying psychovisual framework.

**Limited comparison with related work:** Lin et al. (2023), identified as "the closest work to ours" (line 138), applies Deep Frequency Filtering (DFF) to achieve state-of-the-art results on multiple domain generalization tasks including closed-set classification and open-set retrieval. In contrast, this paper only demonstrates improvements over the obsolete ResNet architecture on a single task (classification), making its contributions appear more limited in scope and impact.

**Imbalanced architectural focus:** Phasor Blocks are introduced to replace a subset of RestNet's spatial layers to break the symmetry of the Fourier Transform (FT). Specifically, Phasor Blocks augment real-valued spatial features with complementary complex-valued ones. They receive disproportionate attention in the main text. Meanwhile, DWConv Blocks used for ConvNeXt adaptation are entirely absent from the main paper discussion. The emphasis on ResNet, rather than modern networks like ConvNeXt, further limits the work's relevance.

**Incomplete resolution of stated objectives:** The limitations section explicitly states: "The key limitation of our work is that though we show SVC organises and encodes selections of object components, it is not clear how these representations are used for reasoning". Since understanding how frequency domain representations enable reasoning was presented as a primary motivation (lines 88, 92), the core contributions remain ambiguous.

**Minor:**
- Emphasis is put on the number of layers in PsychoNet models being lower than their respective baselines, yet parameter counts remain similar.
- Line 451's claim that "This pipeline mimics intermediate abstractions used by the brain to separate feature extraction from higher cognition" requires neuroscience citations to support the biological plausibility.

### Questions
See weaknesses section above. Also:

- ResNet270 shows worse performance than ResNet152 on the same datasets, which is unexpected. Do you have a possible explanation?
- Figure 7 lacks interpretation of why imaginary features activate on whole objects rather than object parts like real features.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces a new biologically-inspired algorithm for augmenting existing CNN-based vision models — PsychoNet. Briefly, this method enables models to translate low-level spatial information into high-level semantic information in the frequency domains using a suite of tools primarily based on 2-dimensional FFT. One key promise of this approach is the ability to represent semantic information in a more global, biologically-rooted way, in the frequency domain.
In general, I quite enjoyed this paper and felt like it was quite ‘dense’ in many respects. While the authors do a good job of laying out the core ideas, I think the ICLR audience would benefit from more scaffolding particularly on the human vision topics. Another consequence of the density of information is that several key ideas and details are bundled in the supplemental information section rather than the main text. 
Overall, I think with some more scaffolded exploration of the background literature with some additional analyses I suggest below. This has the potential to be a strong contribution to ICLR.

### Strengths
* A clear novel architecture deeply grounded in studies on biological vision
* Clear technical explanations of the modeling choices, consequences of ablations, and also some cost-benefit tradeoffs w.r.t. compute in the supplementary materials.
* I appreciate the focus on building better models that aren't primarily based on scaling datasets and using standard transformer-based models.
* Potentially useful for downstream applications in the realm of human/primate vision.

### Weaknesses
* While I generally find the saliency map-based findings compelling in showing that PsychoNets acquire semantic information more efficiently and earlier relative to CNNs, I found the lack of non-accuracy based empirical comparisons lacking. 
For example, if a key claim is that PsychoNet is more aligned with human vision, we should expect it to be more aligned with humans on key failure cases for CNNs including shape bias judgements, and the actual frequency code representations should also be predictive of human neural responses (say on open fMRI datasets like THINGS, etc.)
I think some experiments clearly laying out the contributions of this modeling approach beyond visualizations and accuracy is needed for this to be a valuable contribution for the field.
One could imagine reporting the effects of the ablations currently presented (Fig 6) on these related benchmarks.
* There needs to be more background on psychovisual codes, not just on metrics used to capture these codes, especially given the audience. I think unpacking some of the ideas from Saadane et al., 1998 might be sufficient.
* Figure 5 a should be presented in a larger resolution with clearer font
* While from visual inspection it does appear that models do learn to recognize semantic parts, again having grounded metrics, with respect to annotations might be valuable.

### Questions
N/A. Refer to earlier sections!

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This work introduces Semantic Visual Coding (SVC), a learn frequency domain representation that introduces explicit psychovisual abstraction into CNNs The introduction of SVC is motivated from the perspective of producing disentangled representations to provide a more natural foundation for structured reasoning. SVC works by learning band-limited filters that encode semantics as distinct regions of the Discrete Fourier Transform. SVC is incrporated into some well-known CNN architectures and compared to its standard counter part. Activation maps and filter are visualized.

### Strengths
1. Nice illustration that visualize the methodology and results
2. Interesting link to biologically inspired vision.

### Weaknesses
1. The contribution of the work is unclear. There seems to exist a great deal of works that focus on transforming latent representations into the frequency domain [1, 2, 3, 4]. Some of these works are mentioned, other are not. However, the difference between SVC and existing works is not properly explained, and it is unclear what the methodological contributions are.

2. The experiments are poorly motivated and not properly evaluated. 

(a) It is unclear what the purpose of the image classification experiments in Table 1 is. I interpret this sentence "Since we hypothesise that SVC should handle high-level processing, we stop increasing Phasor Blocks depth after Psycho-B/ResNet-101 to see if it can replace the role of late spatial layers (the width of existing layers are increased to compensate for parameter size.)" as the motivation for the reduction in layers. But the the motivation for the chosen baseline networks is unclear. The ResNet152 and 270 are rarely used, the ResNet18 and and resNet50 are much more common. The poor performance of ResNet152 and 270 on CIFAR10 is most likely due to low amount of samples compared to parameters. For the ConvNext-S, the performance difference is unclear.

(b) The activation maps and filter visualization are nice, but the analysis is highly qualitative. Without any baselines to compare against or quantitative measures, it is unclear what these results are actually demonstrating.

3. Comparison to existing works is missing. Without any comparison to other works, it it is difficult to asses the usefulness of SVC. There seems to be many alternatives that could be used or adapted for comparison [1, 2, 3, 4]. 

- [1] Lin et al., Deep Frequency Filtering for Domain Generalization, CVPR 2023
- [2] Chi et al., Fast Fourier Convolution, NeurIPS 2020
- [3] Rao et al., Global Filter Networks for Image Classification, Neurips 2021
- [4] Huang et al., Adaptive Frequency Filters As Efficient Global Token Mixers, ICCV 2023

### Questions
1. Concretely, in what ways does SVC differ from existing works in the literature like [1, 2, 3, 4]?
2. What benefits does SVC bring compared to existing works?
3. What quantitative measure can illustrate these benefits?
4. Can these benefits be shown experimentally?

- [1] Lin et al., Deep Frequency Filtering for Domain Generalization, CVPR 2023
- [2] Chi et al., Fast Fourier Convolution, NeurIPS 2020
- [3] Rao et al., Global Filter Networks for Image Classification, Neurips 2021
- [4] Huang et al., Adaptive Frequency Filters As Efficient Global Token Mixers, ICCV 2023

### Soundness
2

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
3

### Summary
This paper presents a two-stage hybrid architecture for computer vision that separates low-level feature extraction from high-level abstract reasoning. The first stage employs spatial layers to identify semantically meaningful object parts, augmenting real-valued features with complex-valued counterparts. The second stage transforms these features into the frequency domain using a Fast Fourier Transform, where a dedicated module performs the final classification using learned, sparse, band-limited filters. This architectural approach differs from some prior works that use the frequency domain to improve the efficiency of spatial convolutions or as an integrated global mixer. Here, the frequency domain is used as a distinct final stage to replace the deep spatial layers typically responsible for high-level reasoning. The entire framework is end-to-end differentiable, as all its components, including the Fourier transform and complex-valued convolutions, have well-defined gradients that allow for standard backpropagation-based training. According to the results, the model demonstrates improved interpretability and a reduced dependency on layer depth, though these benefits are accompanied by marginal performance gains over established baselines and a significantly higher computational cost. The work's contribution is therefore positioned as an exploration of an alternative, more transparent model design that bridges spatial feature extraction with frequency-domain abstraction.

### Strengths
The strengths of this work are centered on its architectural design, which aims to improve model interpretability, and the systematic experiments conducted to support its claims. The paper provides qualitative evidence through activation map visualizations suggesting that the framework successfully separates processing stages: early spatial layers learn to identify distinct, semantically meaningful object parts, while the subsequent frequency-domain module performs abstraction and reasoning on these parts. This separation is presented as a more transparent alternative to the entangled computations in deep, homogeneous CNNs.

I think:
1. The framework is explicitly designed to create a more interpretable processing pipeline by separating low-level feature extraction in the spatial domain from high-level reasoning in the frequency domain.

2. The results indicate that the proposed models can achieve comparable or slightly improved performance with significantly fewer layers than their deep ResNet baselines, suggesting that the frequency-domain module effectively handles the high-level processing that would otherwise require additional spatial layers.

3. The paper introduces the Phasor Block, a component whose design can be considered a notable contribution. Instead of adopting a computationally expensive, fully complex-valued network, the Phasor Block serves as a lightweight module that uses standard real-valued operations to generate complementary imaginary features just before they are needed for the Fourier transform. This represents a practical engineering solution to enable more expressive frequency-domain filtering without the full overhead of a complex-valued architecture.
4. The architecture is somewhat grounded in principles of psychovisual processing, providing a clear theoretical motivation for its design choices, particularly the use of coronal frequency bands for semantic abstraction.
5. The authors conduct extensive ablation studies to isolate and validate the contributions of their proposed components, such as the Phasor Blocks and Spectral Branches, adding rigor to their architectural claims.

I think the interpretability is pretty cool, albeit only justified qualitatively.

### Weaknesses
Despite its strengths in interpretability and design, the work has several weaknesses, primarily related to practical applicability and architectural complexity. The most significant drawback is the trade-off between computational cost and performance. The proposed models incur a substantial increase in computational overhead (FLOPs) compared to their baseline counterparts, yet the resulting improvements in classification accuracy are marginal at best, and in some cases, performance slightly degrades. This unfavorable trade-off makes the framework less compelling for applications where efficiency and predictive power are the primary concerns.

1. The models require significantly more FLOPs than the baselines they are compared against. The authors attribute this to the need for higher-resolution feature maps to support the frequency analysis and the use of complex-valued operations that are not highly optimized in current deep learning libraries, which poses a serious barrier to practical deployment.
2. The reported improvements in top-1 accuracy on benchmark datasets like ImageNet are minimal, often less than half a percentage point. Given the large increase in computational requirements, these small gains do not present a strong case for adopting the architecture based on performance alone.
3. Unlike modern architectures such as ResNet and Vision Transformers, which benefit from the simplicity and scalability of stacking homogenous blocks, the proposed framework is a heterogeneous, multi-stage pipeline. This design introduces significant architectural complexity by combining standard convolutional layers, specialized Phasor Blocks, a non-parametric FFT step, and frequency-domain filtering modules. This complexity makes the model less straightforward to scale and modify compared to simply adding more identical blocks.
4. The core concepts leveraged in the paper, such as frequency-domain analysis, complex-valued networks, and biologically-inspired architectures, are all pre-existing areas of research. The contribution can therefore be viewed as a specific and thoughtful synthesis of these ideas rather than the introduction of a fundamentally new technique, which may limit its perceived impact in a crowded field.
5. The framework is exclusively evaluated on image classification tasks. Its effectiveness on other critical computer vision tasks that require dense spatial predictions, such as object detection or semantic segmentation, remains unevaluated. It is unclear how the proposed abstraction in the frequency domain would perform on tasks where preserving precise spatial information is paramount.

Overall, my concerns can be broadly divided into two classes -- first being the improvement relative to FLOP, the second being novelty. Hopefully the authors can provide a more extensive justification.

### Questions
1. You propose the Phasor Blocks for introducing complex-valued features from real-valued inputs. However, their specific internal architecture is not fully justified.

-- What was the design process for the Phasor Blocks? Did you experiment with alternative, perhaps simpler, methods for generating imaginary components, such as a basic 1x1 convolution to project features into a complex space?

-- Why was the specific combination of depthwise and pointwise convolutions chosen? The paper states it's to encourage cross-channel interactions without interfering with spatial relationships, but it would be helpful to see an ablation study comparing this design to other methods.

2. The framework replaces the later stages of a CNN with the frequency-domain pipeline. This choice seems critical to the entire hypothesis but is based on empirical results. Is there a more principled way to determine the optimal depth for this transition, or is it purely a hyperparameter to be tuned for each base architecture?

3. The SVC module partitions the frequency spectrum into three fixed, disjoint radial bands. This seems to contradict the goal of a fully data-driven representation. Would a learnable partitioning scheme, where the model could adapt the frequency band boundaries, lead to better performance or even more specialized filters?

4. The paper repeatedly claims that the SVC module performs "high-level processing and reasoning." However, the evidence shows that it learns to encode selections of object parts. What reasoning is there?

5. The evaluation is confined to image classification on ResNet and ConvNeXt backbones. How do you expect this architecture to perform on dense prediction tasks like semantic segmentation or object detection, where precise, high-resolution spatial information is crucial for the final output? The frequency-domain abstraction inherently discards some spatial localization.

### Soundness
3

### Presentation
3

### Contribution
2

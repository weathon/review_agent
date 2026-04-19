# Minimalist and High-Performance Semantic Segmentation with Plain Vision Transformers

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
In the wake of Masked Image Modeling (MIM), a diverse range of plain, non-hierarchical Vision Transformer (ViT) models have been pre-trained with extensive datasets, offering new paradigms and significant potential for semantic segmentation. Current state-of-the-art systems incorporate numerous inductive biases and employ cumbersome decoders. Building upon the original motivations of plain ViTs, which are simplicity and generality, we explore high-performance 'minimalist' systems to this end. Our primary purpose is to provide simple and efficient baselines for practical semantic segmentation with plain ViTs. Specifically, we first explore the feasibility and methodology for achieving high-performance semantic segmentation using the last feature map. As a result, we introduce the PlainSeg, a model comprising only three 3$\times$3 convolutions in addition to the transformer layers (either encoder or decoder). In this process, we offer insights into two underlying principles: (i) high-resolution features are crucial to high performance in spite of employing simple up-sampling techniques and (ii) the slim transformer decoder requires a much larger learning rate than the wide transformer decoder. On this basis, we further present the PlainSeg-Hier, which allows for the utilization of hierarchical features. Extensive experiments on four popular benchmarks demonstrate the high performance and efficiency of our methods. They can also serve as powerful tools for assessing the transfer ability of base models in semantic segmentation. The codes will be available.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this work, plain vision transformers are used to explore a minimalist system for semantic segmentation. Two models are introduced: PlainSeg and PlainSeg-Hier. PlainSeg uses the last feature map only, while PlainSeg-Hier uses the feature hierarchy. In order to achieve a minimalist system, both models use a Mask2Former transformer decoder that omits the pixel decoder portion. Experiments are conducted on four popular datasets.

### Strengths
This work's writing is understandable and straightforward.

### Weaknesses
The approach makes a relatively small contribution. 
- There are two so-called "principles" presented. Nonetheless, the first one—that high-resolution features are essential for semantic segmentation—has received much research and has long been accepted in the field as conventional knowledge. Regarding the second, even in the CNN era, a common training strategy is to use a large learning rate for the newly initialized decoder.
- In terms of the decoder design itself, the core function of the decoder makes no difference from that of Mask2Former. It is in fact a trivial work to adapt an existing decoder design to a shared backbone network, i.e., adapting Mask2Former decoder to plain ViT in this work.
- It makes a difference to divide a shared feature map into several features using group convolution in order to consistently attend to learnable class tokens. However, it is merely an investigation and will not result in a novel contribution.
- There is no substantial contribution from a changing width configuration in the decoder (Sec. 3.3). The deformable transformer layer used in PlainSeg-Hier comes from existing works (Sec. 3.5).
- The experimental experiments are less persuasive since, according to the author, they did not do fair comparisons and did not pinpoint any particular injustices.

### Questions
- The reviewer believes that ViT-Adapter employs a bottom-up branch since it begins feature aggregation from the input image and concludes with the final feature from the backbone (line 3, paragraph 1, page 2).
- Typo mistake, which shoube be semantic segmentation (line 2, paragraph 2, page 2).
- It would be preferable to display the linear head FLOPs in Table 2 for comparison.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces PlainSeg, a minimalist semantic segmentation system that employs plain Vision Transformers (ViTs) in lieu of more complex, state-of-the-art models. Unlike previous art SegViT, which uses Mask-to-Attention (ATM), PlainSeg incorporates a sequence of three 3x3 convolutional layers. The system demonstrates its efficacy on multiple benchmarks, including the Pascal-context, ADE20K, and COCO-Stuff datasets. The paper argues two main principles: (1) high-resolution features are essential for performance, even when simple up-sampling techniques are used, and (2) a slim transformer decoder demands a significantly higher learning rate compared to a wide transformer decoder. The authors also introduce an extension, PlainSeg-Hier, designed to utilize hierarchical features. Overall, the proposed approach not only achieves competitive performance but also serves as an efficient baseline for semantic segmentation

The proposed approach achieved the best results in the Pascal-context dataset, and it has shown strong performance over its own selected baselines. The datasets are ADE20K and COCO-Stuff 10K/164K.

### Strengths
The proposed PlainSeg model excels in its simplicity and reproducibility, offering a straightforward path for implementation. It has demonstrated robust performance across multiple datasets, including Pascal Context, ADE20K, and COCO-Stuff, thereby validating its practical utility. Key advantages include: 
1) Outperforming SegViT in both efficiency and effectiveness, and 
2) serving as a strong benchmark that can guide future work in semantic segmentation.

### Weaknesses
1. The paper has some shortcomings with respect to its claims of novelty. Specifically, the approach appears to be a minor modification of SegViT and borrows heavily from the foundational Fully Convolutional Networks (FCN). The performance gains over SegViT are also not markedly significant, raising questions about the impact of these changes. 

2. Additionally, the paper's presentation could be improved; it rigidly adheres to a 3x3 convolution followed by Batch Normalization and a 1x1 convolution, a setup akin to that used in FCN. The lack of exploration beyond these hard-coded parameters suggests that the work might be better framed as an experimental report rather than a ICLR paper.

3. While the paper presents extensive experimental results to emphasize the advantages of its simple up-sampling decoder, the approach lacks novelty. Fully Convolutional Networks (FCN) have already established the use of simple up-sampling as a widely recognized baseline in convolutional neural network-based semantic segmentation.

### Questions
1. The paper seems to borrow heavily from Fully Convolutional Networks (FCN), particularly in its use of up-sampling techniques. However, FCN's contributions are not adequately acknowledged or discussed. Could you elaborate on why FCN, a foundational work in semantic segmentation with different backbones like VGG and ResNet, wasn't given its due credit, especially when the proposed approach seems to be an adaptation of FCN with a ViT backbone. Any explanation on this?

2. The paper primarily focuses on datasets that don't require fine-grained semantic segmentation, making me skeptical about the model's capability to handle intricate object boundaries. It would be interesting to have the authors to test the proposed approach on datasets requiring more detailed semantic segmentation, e.g. Pascal VOC? How does the proposed work against the other works with denseCRF or diluted Convolution?

3. A more technical question: when using bilinear interpolation twice in the proposed methods, could authors specify the exact implementation and parameters used? Was the bilinear interpolation learned or pre-defined? Did you use PyTorch's native bilinear interpolation, or some other implementation? Different interpolation techniques can produce different results, it would be interesting to see if this makes a difference on the final results.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a plain ViT-based semantic segmentation approach. Unlike many other state-of-the-art approaches that create some form of hierarchical ViT to utilize the hierarchical features for the segmentation task, the proposed PlainSeg approach is building a minimalistic and simple decoder on top of a plain ViT model. In essence the paper explores how to create a lightweight variant of the mask2former module that can be attached to a ViT, without the need for hierarchical features. The resulting model performs competitively while at the same time not requiring any architectural modifications of the underlying ViT. In addition to that, a slightly more complicated, but stronger PlainSeg-Hier is proposed, that does create hierarchical features from the ViT to feed it into the attached segmentation decoder.

### Strengths
- I like the main goal of the paper, to utilize a ViT and put a simple and minimalistic network on top of it in order to do semantic segmentation. The paper shows that this is possible while remaining competitive and knowing that a simpler model is an option is always good for future research.
- In direct comparison to ViT-Adapter, the proposed method seems to perform on par, while requiring fewer additional parameters in the decoder and being computationally more efficient. In comparison with SegViT the model performs better, albeit it is a bit slower.
- The ablations regarding the learning rate reveal some interesting effects.

### Weaknesses
- What is the main take-away message and contribution? As I said before, I'm happy to know a simple decoder can be put on top of a plain ViT, but how simple is this decoder really? The initial decoding of feature maps with convs is of course trivial, but what follows seems to be very similar to a full-fledged Mask2Former with minor simplifications. All of a sudden, it doesn't feel so simple anymore. How was this model trained? Using the standard Mask2Former recipe, including Hungarian matching and masked cross-attention? Even if we disregard how simple or complex the decoder is, it's still a vanilla ViT and that is interesting, but this is also where I feel that the novelty ends. It would have been more valuable if we would gain some insights why exactly this architecture is good or should be chosen, but the paper here just shows that with slight modifications an existing decoder can be put on top of a plain ViT. As such it feels a bit more like an engineering paper and there is little real novel insights to be gained.

- Partially the paper is hard to follow and some stuff is simply not described very well, but it's rather assumed the reader just knows. For example:
  - The short section on PlainSeg-Hier is very confusing and to me this architecture still isn't completely clear. Even with the additional explanation in the appendix, I can't say I for sure I now know how this network looks like.
  - Sometimes there are missing citations, for example if you write that "Layer-wise learning rate decay (LLRD) has been widely adopted", please give some citations for that. Which "sliding window strategy" do you use? Or when you write "On this basis, we further present PlainSeg-Hier which incorporates hierarchical features, as seen in previous works." which works do you mean?
  - At some point an ablation writes about "width-to-depth", while it's not clear what this really means
  - As stated above, how closely is the training process aligned to that of Mask2Former and where does it diverge?
  - What are the feature map visualizations? Magnitudes of the features? PCA projections? Random channels?

- For some of the tables I'm unclear where the baseline numbers come from. For example for Table 3 I can't find the respective numbers in the ViT-Adapater paper. Even for settings directly reported in the paper, e.g. ADE20k results, I can't find a setup with 568M parameters getting a 58.32 mIoU, so how were these numbers obtained? And how representative are they for the original approaches?

- In a certain way, the Segment Anything Model (SAM) also uses a simple ViT and a super small decoder on top. I know that SAM is promptable and it's not really great at spitting out semantic segmentations, nevertheless the core idea is rather similar and it should be discussed in related work.

### Questions
- Large part of the paper is about PlainSeg, but the short section on PlainSeg-Hier introduces another variant that is mostly used as the main contender in the tables. What is the real value of PlainSeg and why not discuss PlainSeg-Hier in more detail?

- Why the strong focus on MIM? Couldn't other pretraining methods such as DINO(v2) work too?

- You write that you use deconvolutions. I thought by now this term has vanished and nobody uses it anymore. Please use transposed convolutions instead, since deconvolutions are something completely different!

- Equation 2 has an additional Norm that is missing from Figure 1. Which of these is correct?

- There are quite some weird terms and typos, e.g. "to render efficacy", although maybe my English just isn't that great, but also "objective detection", "induction bias" and "multi-model data". Maybe run this through a grammar/spell checker.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper focuses on the problem of semantic image segmentation. The author proposes a high-performance ‘minimalist’  segmentation system based on plain ViTs. Meanwhile, the paper also offers some insights into the practical principles for adapting plain ViTs to semantic segmentation task. The experimental evaluations are conducted on several popular semantic image segmentation datasets and the experimental results look good. It achieves highly competitive performance compared to ViT-Adapter and outperforms SegViT.

### Strengths
1. The experimental evaluations are conducted on four popular and challenging semantic image segmentation datasets ADE20K, Pascal Context, COCO-Stuff 10K, and COCO-Stuff 164K, and the experimental results (accuracy, parameters, efficiency) seem good when compared to previous high-performance solutions.
2. Overall, the writing and illustrations are clear and easy to follow.

### Weaknesses
1. The major concern of the whole paper is its technical novelty. The author claims two findings and principles: (1) high-resolution features are important for high performance (2) the slim transformer decoder requires a much larger learning rate than the wide one. While the first point is obvious and has already been mentioned by various previous algorithms, and the second point could be a training trick, rather than something inspirational. Combining plain ViTs with a hierarchical structure is not new, hierarchical design already exists in many other frameworks, and the whole pipeline in Figure 1 is not inspiring and appealing, and is hard to motivate potential readers.
2. Some of the content and paper organization need to be adjusted. For example, in the introduction section, the former two paragraphs are more like to be a paragraph in the related work section. While it emphasizes more on the motivation of the work. In Table 3, some lower performance results are highlighted, it is something stranger and needs to be modified.

### Questions
In-depth analysis and inspirational designs are needed to make the whole paper and algorithm more appealing, and thus can motivate potential readers.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

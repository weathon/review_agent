# Light of Normals: Unified Feature Representation for Universal Photometric Stereo

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 2, 8, 8

## Abstract
Universal photometric stereo (PS) is defined by two factors: it must (i) operate under arbitrary, unknown lighting conditions and (ii) avoid reliance on specific illumination models. Despite progress (e.g., SDM UniPS), two challenges remain. First, current encoders cannot guarantee that illumination and normal information are decoupled. To enforce decoupling, we introduce LINO UniPS with two key components: (i) Light Register Tokens with light alignment supervision to aggregate point, direction, and environment lights; (ii) Interleaved Attention Block featuring global cross-image attention that takes all lighting conditions together so the encoder can factor out lighting while retaining normal-related evidence. Second, high-frequency geometric details are easily lost. We address this with (i) a Wavelet-based Dual-branch Architecture and (ii) a Normal-gradient Perception Loss. These techniques yield a \textbf{unified} feature space in which lighting is explicitly represented by register tokens, while normal details are preserved via wavelet branch. We further introduce PS-Verse, a large-scale synthetic dataset graded by geometric complexity and lighting diversity, and adopt curriculum training from simple to complex scenes. Extensive experiments show new state-of-the-art results on public benchmarks (e.g., DiLiGenT, Luces), stronger generalization to real materials, and improved efficiency; ablations confirm that Light Register Tokens + Interleaved Attention Block drive better feature decoupling, while Wavelet-based Dual-branch Architecture + Normal-gradient Perception Loss recover finer details.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper proposes LINO UniPS, a transformer-based framework for Universal Photometric Stereo. It recovers surface normal maps from multiple images under unknown lighting conditions. They propose the Light Register Tokens and the Interleaved Attention Block to allow the model to understand normal and light conditions separately. Also, they propose the Wavelet-based Dual-branch Architecture and a normal-gradient perception loss to preserve high-frequency geometric details. They demonstrated their methods on PS-Verse, a new, high-quality, large-scale synthetic dataset they created.

### Strengths
- The explicit light-feature decoupling through LRTs and interleaved attention is well justified and ablated.
- Consistent performance improvements have been achieved, including in the benchmarks presented.
- The paper is well-organized.

### Weaknesses
- Some ablations (e.g., the effect of each light type token separately) could be presented in more detail. 
- There are no qualitative results for ablation studies.

### Questions
- Looking at the Light Registered Attention module in Figure 2, it seems like frame tokens are copied and utilized for each of the three light registers. Am I understanding this correctly? This part seems like a huge computational burden. Why did you bother calculating the three separately?
- You claim that each of your methods solves the decomposing and high-frequency problems, but can you show qualitative ablation, not just performance improvement?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes LINO UniPS, a method for universal photometric stereo. It first introduces light register tokens for different types of lightings and interleaved attention block to enable better separation of lighting and lighting-invariant normal in the encoder. Then it uses a Wavelet-based Dual-branch architecture and a normal-gradient perception loss to recover the the fine details of the scenes. Furthermore, it introduces a new dataset, PS-Verse, with more complicated surface and lightings for the photometric stereo task.

### Strengths
1. The proposed method achieves good results on two benchmark datasets, DiLiGenT and LUCES.

2. The paper introduces a new dataset, PS-Verse, which is proved to be able to help achieve better performance for the same method from Table 2.

### Weaknesses
1. The clarity of the paper could be further improved. In general I can understand the idea of the paper but there are several key aspects that I am confused with. Please see Questions section below. Also there are not metrics introduction for table 4 and table 5 in their table descriptions.

2. The test scenes only have one object per-scene. I wonder if the method can handle more complicated scenes?

3. My biggest concern is the comparison results with other methods. 

(1) In Uni MS-PS (https://hal.science/hal-04431103v2/file/main_hal.pdf) Table 3, Uni MS-PS can achieve 6.04 when using 30 images for Buddha, which is better than LINO UniPS and not shown in Table 4. For POT1, Uni MS-PS can achieve 4.08 which is very similar to LINO UniPS. For COW, best Uni MS-PS is the same as LINO UniPS and also not shown in table 4.

(2) The same is for LUCES dataset, not the best results of Uni MS-PS are reported in table 5, e.g. for BOWL, BUDDHA.

### Questions
1. The training starts from PS-Verse Level 1 data to level 4 data, which don't have ground-truth normal maps, I wonder how do you compute $L_n$ for these data?

2. For the feature similarity metric (CISM), what are the two features to compare with?

3. In table 4, for POT2, SDM-UniPS seems to achieve better results?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper proposes a new encoder to address two issues in Photometric Stereo (PS): (1) the coupling between normal and lighting information, and (2) the loss of high-frequency details. To tackle the first problem, the authors introduce Light Register Tokens and an Interleaved Attention Block to decouple normal and lighting features. Furthermore, they employ a wavelet-based dual-branch structure combined with specific loss to preserve high-frequency details.

### Strengths
The proposed method demonstrates solid performance on both real and synthetic datasets. The ablation studies are comprehensive and well-conducted.

### Weaknesses
- Line 64 seems quite contradictory, should it be normal features instead?
- Could the authors provide a more physically grounded explanation for explaining the effect of feature similarity? The current discussion mainly relies on experiments illustration.
- Could the Light Token also be used for light source estimation? If so, how accurate would it be?
- It would be much better to discuss and acknowledge PS work under general setup.

### Questions
See above

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
In this paper, the authors introduce a ViT-based framework, named LINO UniPS, for universal photometric stereo. They target at tackling two major challeneges in universal photometric stereo, namely 1) ineffective decoupling of illumination and normal cues and 2) loss of geometry details. Specifically, they introduce Light Register Tokens and an Interleaved Attention Block to explicitly decouple illumination from normal features, yielding a unified feature representation. They also adopt a Wavelet-based Dual-branch Architecture and a Normal-gradient Perception Loss that substantially improve the reconstruction of fine-grained geometric details. Further, they build a synthetic dataset, named PS-Verse, graded with surface complexity and lighting diversity to support similar research.

### Strengths
+ The introduction of three types of Light Register Tokens for aggregration of illumination information of three different illumination types sounds logical and novel. It helps improve the decoupling of illumination from normal features. Its effectiveness has been demonstrarted in ablation study.
+ The proposed light alignment supervision sounds logical and novel. It helps Light Register Tokens learn to capture the respective illumination information. Its effectiveness has been demonstrated in ablation study.
+ The Interleaved Attention Block introduces a global cross-image attention mechanism. Although global attention is not a novel idea, but the four interleaved attention layers allow aggregrating information across multiple hierarchical levels and help better decoupling illumination and normal features. The effectiveness of the Interleaved Attention Block has been demonstrated in ablation study.
+ The Wavelet-based Dual-Branch Architecture sounds logical and novel. It helps preserve details throughout the network. Its effectiveness has been demonstrated in ablation study.
+ The Normal Gradient Perception Loss sounds logical and novel. It helps to enhance high-frequency areas. Its effectiveness has been demonstrated in ablation study.
+ The large-scale synthetic dataset PS-Verse can benefit further research in universal photometric stereo.
+ SOTA results have been reported on DiLiGenT and Luces datasets.

### Weaknesses
- The figures and captions need further improvement. For instance, the pipeline in fig. 2 is rather complicated and difficult to understand. It does not match well with the detailed description of the modules. What do the different colors represent? In fig. 3, it is not clear how to interpret the attention maps for the different Light Register Tokens. More detailed discussions are needed to better understand how the figures demonstrate the effectiveness of the different Light Register Tokens.    
- The concept of inter-image and intra-image context in this paper is rather confusing. In lines 228-229, it mentioned that frame attention captures inter-image context. Should the frame attention capture intra-image (within-image) context while the global attention capture inter-image (between-images) context instead? Discussions in the Appendix also show the same confusion.
- The ablation study has demonstrated the effectiveness of each proposed component quantitatively. It would, however, be also important to show the corresponding qualitative results to enable readers to visually perceive the effecti of each component.
- Even with the details provided in the Appendix, it is not sufficient to reproduce the results. For instance, in lines 707-714, the Light Register Tokens have a dimension of C, but they are concatenated with the wavelet/downsample tokens with a dimension of D. In lines 751-755, should the "1 - " goes inside the summation sign? Otherwsie, what is the purpose of including "1" in the loss? In lines 808-814, the steps in transformating the aggregated features into the four-level feature pyramid is not very clear.

### Questions
- Theoretically, the wavelet transform already includes a downsampled version of the image. Why is it necessary to include a naive downsampling branch in parallel?  
- Three Light Register Tokens are introduced specifically for three illumination types, namely point, direction, and environment lights. Have the authors consider using a single generic light register token instead? Does this design generalize well to other light representations?

### Soundness
4

### Presentation
3

### Contribution
4

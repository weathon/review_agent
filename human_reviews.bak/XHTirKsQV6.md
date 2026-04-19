# ProtoSnap: Prototype Alignment For Cuneiform Signs

- Decision: Accept (Poster)
- Scores: 8, 6, 6, 5

## Abstract
The cuneiform writing system served as the medium for transmitting knowledge
in the ancient Near East for a period of over three thousand years. Cuneiform
signs have a complex internal structure which is the subject of expert paleographic
analysis, as variations in sign shapes bear witness to historical developments and
transmission of writing and culture over time. However, prior automated techniques
mostly treat sign types as categorical and do not explicitly model their highly varied
internal configurations. In this work, we present an unsupervised approach for
recovering the fine-grained internal configuration of cuneiform signs by leveraging
powerful generative models and the appearance and structure of prototype font
images as priors. Our approach, ProtoSnap, enforces structural consistency on
matches found with deep image features to estimate the diverse configurations
of cuneiform characters, snapping a skeleton-based template to photographed
cuneiform signs. We provide a new benchmark of expert annotations and evaluate
our method on this task. Our evaluation shows that our approach succeeds in
aligning prototype skeletons to a wide variety of cuneiform signs. Moreover, we
show that conditioning on structures produced by our method allows for generating
synthetic data with correct structural configurations, significantly boosting the
performance of cuneiform sign recognition beyond existing techniques, in particular
over rare signs. Our code, data, and trained models are available at the project page:
https://tau-vailab.github.io/ProtoSnap/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper presents a method for aligning cuneiform character prototypes to in-the-wild real character images. The prototype consists of an image of the canonical character, as well as an aligned skeleton representation of the character. The method uses deep image features from a finetuned diffusion generation model to measure the patch similarities between the prototype image and the real image. Given the similarity map, the method first applies a global affine transformation to the skeletal representation, such that the feature similarities are maximized. Then the method applies per-stroke projective transformation to the skeleton strokes, to further maximize the alignment of image features. To regularize the alignment optimization, mutual optimal matching between patches, RANSAC for global transform, as well as saliency, identity and boundary constraints for local transform are applied. 

The method has been tested on a new benchmarked collected by expert annotators, and showed improved accuracy than baseline matching algorithms. The aligned image-character pairs also allow for finetuning ControlNet, so that new images can be synthesized for training OCR models, which demonstrates the benefits enabled by the method.

Overall, this paper presents a fluent combination of various image processing and registration tools to solve the problem of cuneiform character recognition to a better state.

### Strengths
The paper is well written and well illustrated. Technical designs are presented concisely in the main text and discussed in detail in the appendix. Experiments are extensive, using a new benchmark with images labeled by experts and crowdsourcing. Ablation studies are exhaustive and confirmative of the various technical designs. The use of aligned prototypes for generative data synthesis is particularly interesting, by bridging the power of pretrained ControlNet and the parameterized cuneiform images.

### Weaknesses
The main weakness is that only cuneiform characters are considered. The authors did not discuss if the same set of techniques used by the method pipeline can be applied to other types of characters, like oracles. It's desirable to at least discuss the possibilities and challenges, e.g. within the different skeleton structures and permitted deformations.

More analysis of the collected and training datasets can be done, to provide the readers with more understanding of the common signs and variations.

### Questions
As mentioned above, I hope the authors can discuss the extension and challenges when applying the method to more types of ancient characters. In particular, oracle bone characters have different organizations than cuneiforms, as an oracle bone character not only consists of multiple strokes, but more importantly the strokes are not fixed in numbers and shapes as the cuneiforms do; instead, the strokes are connected into components depicting certain figures, which can vary to a large extent in terms of the number of strokes and nonrigid deformation. To handle such variations, the assumed projective per-stroke transform may not be sufficient. In particular:
1. Are there any components of the current method that would need significant modification?
2. Could the authors provide a brief analysis of how the skeleton structure and deformation models need to be adapted?

Fig.6 can be more specific about "the most prevalent sign variation during training". Which are these variants and how do you know it? Specifically, the authors can provide more insights into the dataset, by the following means:
1. Provide quantitative data on the frequency of different sign variants in their training set, if available.
2. Explain the method used to determine which variants were most prevalent.
3. Include a brief discussion about how this prevalence might impact the model's performance.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper present a method for fine-grained structure retrieval in cuneiform sing images, given the canonical form of the depicted signs. In more detail, the method first calculates a global alignment using diffusion features, best buddies correspondences and RANSAC. Then,  the aligned template sing is refined via skeleton-based optimization. The authors demonstrate sota results in cuneiform sign alignment and recognition. Last, a new dataset of expert-annotated cuneiform sign images and will be released.

### Strengths
1. The authors have done a good job presenting their method to a reader unfamiliar with the subject. The paper is well written and the ideas well presented.
2. The method is novel for cuneiform sign alignment, as it adopts a common tactic from pose/keypoint detection problems in the scope of the presented subject.
3. The method achieves sota results in cuneiform sign alignment, although a more detailed comparisons scheme could have been designed (more details in weaknesses 1). 
4. The method achieves sota results in cuneiform sign recognition.
5. A new benchmark dataset of cuneiform sign images with expert annotations will be released.

### Weaknesses
1. Comparisons in Table 1 are not clear. To my understanding, for DINOv2 and DIFT the authors directly decide keypoints based on feature similarity without solving RANSAC. On the other hand, the authors employ RANSAC for SIFT features and their method (with or without refinement). In my view, the authors should not focus on a single model for feature extraction, but rather experiment with all of them in the same setting (with or without RANSAC) and present their method as a more general method for template alignment for cuneiform signs.

### Questions
1. I kindly ask the authors to provide more details and motivation regarding experimental results in Table 1.
2. To my understanding, cuneiform sign data are few if not limited. How far away is the field from data-driven methods that predict accurate keypoints on sign images without known templates? Could your synthetic data be helpful towards this direction?

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
3

### Summary
This paper introduces ProtoSnap, an unsupervised method for aligning the internal structure of cuneiform signs using generative models and prototype images. The approach improves the recognition of cuneiform signs by refining the alignment of skeletal templates to real sign images. This method leverages deep learning and generative modeling to interpret the complex internal configurations of cuneiform signs, enhancing optical character recognition (OCR) accuracy, especially for rare signs.

### Strengths
+ The application of unsupervised learning and prototype alignment to cuneiform signs is novel and shows significant potential.

+ The technical approach is sound, utilizing SoTA techniques in image processing and machine learning.

+ The method has clear applications in digital humanities, aiding the decipherment and study of ancient texts.

### Weaknesses
1. There’s a potential risk that the method could overfit to the prototypes it has been trained on, especially if those prototypes do not capture the full variability of the signs in the dataset.

2. It would be good if the authors could report on the computational resources required for implementing the ProtoSnap method. Considering that it involves deep learning models and generative processes for aligning prototypes with actual images, understanding the computational demands is crucial.

3. I'd like to hear the authors' opinion on the potential of ProtoSnap to adapt to other ancient scripts, which often present unique challenges in terms of symbol complexity and degradation patterns. This discussion could provide valuable insights into the versatility and scalability of the proposed method beyond cuneiform studies.

### Questions
1. Can the authors detail any specific preprocessing steps required to prepare the cuneiform images before applying ProtoSnap?

2. What are the limitations in terms of computational resources, and how scalable is this approach when applied to large datasets of cuneiform texts?

3. How does ProtoSnap handle extremely degraded or incomplete cuneiform signs where the prototype may not initially align well?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes a novel approach to handle the complex internal structure of cuneiform signs, called ProtoSnap, an unsupervised method that utilizes deep generative models and prototype font images to estimate the fine-grained internal structure of cuneiform signs.

### Strengths
Originality: ProtoSnap's use of deep diffusion features and skeleton - based prototypes for unsupervised cuneiform sign alignment is novel.

Quality: The overall flow of the methodology section is sound and logical.

Clarity: This paper is clearer on the whole, from the introduction part of the cuneiform research background and the limitations of the existing methods, which naturally leads to the research objective, i.e., to propose the ProtoSnap method to solve the problem of analysing the internal structure of the cuneiform symbols.

Significance: This work was instrumental in the development of the field of cuneiform writing.

### Weaknesses
1. While this paper presents a new benchmark for evaluation, the current dataset may not be fully representative of the variety of cuneiform symbol variants and writing conditions present in the historical record.
2. The superiority of the method proposed in this paper is not reflected in the related work.
3. 4D similarity volumes in section 4.1 are not clearly described.
4. While the method shows promise for cuneiform signs, its adaptation to other ancient writing systems or complex symbol sets may not be straightforward.
5. There are too few comparative experiments to adequately demonstrate the superiority of the proposed method.

### Questions
1. "This H ×W×H ×W tensor, visualized in Figure 3, contains the pairwise cosine similarities between features encoding patches of the prototype and target images." I'm confused about the H ×W ×H ×W.

2. The proposal includes a user study to assess the practical usability and effectiveness of the ProtoSnap methodology from the perspective of an end user such as an archaeologist or historian.

3. Recommendations for fuller experiments.

### Soundness
3

### Presentation
2

### Contribution
2

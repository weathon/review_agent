# DC3DO: Diffusion Classifier for 3D Objects

- Decision: Reject
- Scores: 3, 3, 3, 3

## Abstract
Recent advancements in deep generative models, particularly diffusion models, have shown remarkable capabilities in generating high-fidelity 3D objects. In this work, we explore the application of diffusion models for 3D object classification by integrating the LION model with diffusion-based classifiers. Due to the availability of pretrained model weights, our study focuses on two categories from the ShapeNet dataset: chairs and cars. We propose DC3DO, a method that leverages the generative strengths of diffusion models for domain generalization in 3D classification tasks. Our approach demonstrates improved performance over a multi-view baseline, highlighting the potential of diffusion models in handling 3D data. We also examine the model's ability to generalize to data from different distributions, evaluating its performance on the IFCNet and ModelNet datasets. This study underscores the potential of using diffusion models for 3D object classification and sets the stage for future research involving more categories as resources become available.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper explores the use of 3D diffusion models for zero-shot 3D object classification. This paper applies the diffusion classifier (Li et al., 2023a) on a point cloud diffusion model (Zeng et al., 2022a). The experiments show that the proposed method outperforms the baseline, which applies the diffusion classifier directly on multi-view renderings, and classifies based on voting.

### Strengths
The writing of the introduction looks fine.

### Weaknesses
1. The baseline is not well designed. The original diffusion classifier (Li et al., 2023a) is based on StableDiffusion, which was trained on natural images rather than point cloud rendering. It is expectable that directly applying the diffusion classifier on the point cloud rendering gives bad results.
2. The main experiments are not thorough. The model is only tested on cars and chairs in ShapeNet. Seven years ago, PointNet [A] has already set a good benchmark for point cloud classification.
3. The out-of-distribution experiments are barely reasonable. Both the in-distribution and out-of-distribution data are just chairs, though from different datasets, ShapeNet VS ModelNet.
4. Considering the accuracy of classifying chairs is less than 50%, it is not proper to describe the model as "accurate" in multiple places in the paper, such as line 190, 433, 485, and 501.
5. This paper contains useless paragraphs. For example, line 440-443, it says increasing the image size and number of views can slow down the processing time of the multi-view classifider, which is too obvious.
6. The writing is confusing. In the introduction, it says MVDC is just a baseline and DC3DO is the proposed model. However, the whole ablation study is about how the input image size affects the performance and inference time of MVDC.
7. Classifying each object takes approximately 20 seconds.

[A] Charles R. Qi, Hao Su, Kaichun Mo, Leonidas J. Guibas, "PointNet: Deep Learning on Point Sets for 3D Classification and Segmentation", CVPR 2017.

### Questions
What if the multi-view diffusion classifier is fine-tuned on point cloud rendering? Will it significantly change the performance?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes a diffusion-based 3D object classification approach. Specifically, they adapt the LION (diffusion-based model for point cloud to mesh generation) into a classification model that defines P(x|c) where x is the given 3D geometry, and c is the object class. They conduct an experiment on a small subset of the ShapeNet dataset and evaluate it on both ShapeNet and ModelNet. The proposed approach seems reasonable, however, the experiment setup is not sound enough to demonstration the advantages of the proposed approach.

### Strengths
1. The method is straightforward and easy to understand.
2. Exploring the 3D object understanding and classification seems a worth study topic.

### Weaknesses
1. The experiment setup, especially for the baseline MVCNN is confusing, and the accuracy for the baseline seems problematic. According to Line 253, the classification is defined as a close-set classification problem that uses the class given the largest P(x|c) as the prediction, however, in line 309, the baseline is evaluated as a binary classification problem (whether belongs to category car or not). Also, for both 3-class classification and binary classification problem the accuracy of Chiar in Table 1 is lower than random guess (33.3% or 50%), which seems the baseline is totally not working. Thus, the experiment results seem not meaningful.
2. The proposed method is incremental. But compared to weakness 1, this is not a significant weakness.

### Questions
1. For the baseline what is the renderer setup? What texture/illumination is used when rendering the images?
2. Why do only equations on pages 4 and 8 have equation numbers?

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper proposes DC3DO, a pipeline that adopts 2D diffusion types of models to zero-shot 3D object classification. DC3DO is extended from previous work LION that adopts diffusion models to 2D classification model by introducing multi-view diffusion components. The model is tested on ShapeNet (only on chairs and cars categories).

### Strengths
1. The paper is easy to follow, the presentation of the work is good.
2. Figures are clear and easily understandable, and they help reader conceive the main idea the paper is trying to present.
3. Interesting illustrations of certain categories with high and low prediction accuracies of car and chair

### Weaknesses
1. The biggest problem of this paper is the lack of enough experimentation. The experiments are only conducted on two selected categories (car and chairs) from one dataset (ShapeNet), which is definitely not enough for a paper at this conference. Also, not enough baselines are considered, and the paper only compares to baseline MVDC, which is not a recent work. The current experiment results cannot support the claim of the paper.

2. The paper lacks technical novelties. MVDC seems only extend LION to multi-view without much structural and strategical changes. It seems to be a naive multi-view extension of LION. 

3. The paper fails to discuss and tackle the potential drawback of its proposed solution. As I can see, the multi-view diffusion process will take a long time on every image, which will be much slower than some other 3D object classification models. The paper should analyze the tradeoff between performance and efficiency compared to more typical 3D object classification models. Does the performance boost worth the increasing inference time? 

4. The claim that this method is robust is also not sufficiently supported by the experiment. The experiment does not compare to any other baseline at all to show the proposed method is more robust.

### Questions
Please see the weakness session.

I think the paper needs serious revision to include more experimentations and to reduce the inference time overhead.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
This paper proposes a Diffusion Classifier designed for 3D objects (DC3DO). DC3DO combines the LION model with the 2D Diffusion Classifier to build a class-conditioned 3D Diffusion Classifier based on latent encoded from the 3D shape and points of objects. 

The proposed DC3DO is mainly compared with the Multi-View Diffusion Classifier(MVDC), which combines the aggregated multi-view features and the 2D Diffusion Classifier. 

Out-of-distribution performance is evaluated on unseen datasets other than the training dataset.

### Strengths
The idea of building a 3D diffusion classifier is attractive. The paper compared two methods of combining 3D representation and 2D diffusion classifier. 

The paper is well-written and clear to follow.

### Weaknesses
1. Limited number of classes are validated: only “chairs” and “cars” are evaluated for classification performance. Even though the current 3D datasets are relatively smaller than 2D datasets, two categories are insufficient for validating a classifier considering the MVCNN (Su et al., 2015) was validated on 40 classes. 

2. Limited comparison baselines: at least MVCCN is closely related to MVDC in this paper and as a frequently mentioned baseline method, can I know why the authors didn't compare the main classification results with MVCNN? And also other standard baseline 3D classification models are expected.

3. Limited novelty: The main contribution of DC3DO is a simple combination of LION and Diffusion Classifier. Considering the combination is the same way as turning a 2D diffusion model into a classifier as Diffusion Classifier (Li et al., 2023), the novelty of this paper is limited.

4. Misuse of the term “zero-shot classification”: zero-shot classification is supposed to generalize a classifier to unseen "classes". In this paper only 3D models of the same class from unseen datasets/sources are validated as OOD data, it should be best described as a domain generalization instead of a zero-shot classification. Can the author provide further clarification on this?

### Questions
1. Why no more categories are validated for a classifier? Please refer to Weaknesses 1.

2. Why no other baseline method is compared? Please refer to Weaknesses 2. Especially for OOD settings, experiment without comparisons has very limited value.

### Soundness
2

### Presentation
3

### Contribution
2

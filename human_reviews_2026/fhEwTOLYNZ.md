# Designing Affine-Invariant Neural Networks for Photometric Corruption Robustness and Generalization

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 6, 2, 6

## Abstract
Standard Convolutional Neural Networks are notoriously sensitive to photometric variations, a critical flaw that data augmentation only partially mitigates without offering formal guarantees. We introduce the *Scale-Equivariant Shift-Invariant* (*SEqSI*) model, a novel architecture that achieves intensity scale equivariance and intensity shift invariance by design, enabling full invariance to global intensity affine transformations with appropriate post-processing. By strategically prepending a single shift-invariant layer to a scale-equivariant backbone, *SEqSI* provides these formal guarantees while remaining fully compatible with common components like ReLU. We benchmark *SEqSI* against *Standard*, *Scale-Equivariant* (*SEq*), and *Affine-Equivariant* (*AffEq*) models on 2D and 3D image-classification and object-localization tasks. Our experiments demonstrate that *SEqSI* architectural properties provide certified robustness to affine intensity transformations and enhances generalization across non-affine corruptions and domain shifts in challenging real-world applications like biological image analysis. This work establishes *SEqSI* as a practical and principled approach for building photometrically robust models without major trade-offs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces SEqSI, a convolutional network that guarantees invariance to global shifts and equivariance to global scales by design. The proposed SEqSI block ensures equivariance to scale and invariance to shifts by constraining the weights of the kernel to be zero-sum in the first block. Equivariance to scale is maintained through the rest of the network via bias-free networks. Notably, the network is significantly less constrained compared to [Herbreteau, 2023] by only requiring weight constraint at the first layer, instead of throughout the network. This yields a network that is significantly more efficient in terms of training, inference, and memory complexity while maintaining the core functionality and achieving on par or better performance. Furthermore, the paper introduces pipeline to extend the affine-equivariance to thresholding tasks such as object localization. This is achieved by replacing the conventional sigmoidal thresholding with a novel thresholding based on the standard deviation of the entire map, which is provably equivariance, as well as a novel Z-MSE loss function.

On classification tasks, SEqSI is shown to have better performance than baseline models (SEq, AffEq, etc) when no data augmentation is available and performance on par with baselines under augmentation. SEqSI also achieves strong performance to non-affine transformations that it is not specifically designed for, showcasing flexibility. These result showcases the benefits of designing for invariance and equivariance in a convolutional network. 

On object localization tasks SEqSi significantly outperforms baseline methods, including a straight sweep for affine-transformation, both under and without augmentation. It achieves strong performance as well for non-affine transformations. This can be attributed to its novel thresholding scheme that preserves equivariance and invariance throughout the entire transformation.

Overall SEqSI represents an elegant solution to a problem with clear motivations and practical applications, such as biomedical imaging. My initial recommendation is a 6 (weak accept) which should be easily increased to an 8 (strong accept) with a few clarifications from the authors (please see weaknesses and questions section).

### Strengths
* The paper is well motivated and provides an elegant solution.
* SEqSI is provably equivariant to scale and invariant to shift. Furthermore, SEqSI achieves this with significantly less constraint than baseline line methods.
* SEqSI introduces a pipeline and loss function that extends equivariance to thresholding tasks, enabling wider application to tasks such as object localization and binary segmentation.
* SEqSI demonstrates strong performance compared to baseline methods on both classification and object localization task, both with and without data augmentation, and under affine and non-affine transformations.
* SEqSI has computational efficiency significantly better than fully affine-equivariant baselines and on par with standard models without any equivariance and only scale-equivariance models.

### Weaknesses
While I believe this is overall a good paper that meets all ICLR standards for acceptance, I do think there are a few weaknesses that I would like the authors to address during rebuttal.
1. The classification test in Table 3 is done on a relatively small and low resolution dataset (CIFAR-10). It would be more compelling if the authors could also present results on higher resolution (ie Stanford Cars, Oxford Pets, Caltech-101) and larger datasets (ImageNet). I don't think it is reasonable to run all experiments or the ImageNet dataset in the rebuttal time frame. If the authors can showcase results for non-augmented train set under only the four affine transformations for standard, SEqSI, and SEq for datasets such as Oxford Pets, that would convince me the scalability of the model.
2. I think it would be cool to showcase binary segmentation results on a small toy dataset such as Caltech-101, if time permits.
3. I am confused as to why Table 4 only includes non-equivariant baselines? While this demonstrates the benefits of equivariance, I don't think that was ever in question. Filling out the additional baselines would serve the paper much better.
4. I think the authors should spend more time talking about the architectural differences between SEqSI and [Herbreteau, 2023], as they appear very similar on first glance, in terms of what they want to achieve and the ways they go about achieving them.

### Questions
1. The authors mentioned that the proposed thresholding scheme can also be extended to tasks such as foreground-background segmentation. Would this hold for tasks requiring additional classes such as semantic segmentation? This may further extend the use case of the proposed method for situations such as autonomous driving where cameras are often subject to significantly interference with light sources such as the sun or metallic reflection.
2. While not completely related, since 2023 there have been works that deal with perceptual variation in images from the point of view of color equivariance. Specifically [1] first introduced hue-equivariance using Group CNNs [2] by rotating the RGB cube through (1,1,1). [3] achieves this using soft equivariance. [4] extends this idea to cover equivariance to hue, saturation, and luminance. I wonder if it would be worth while to talk about these in related works.

[1] Attila  Lengyel, et al. "Color Equivariant Convolutional Networks." NeurIPS 2023.\
[2] Taco Cohen, and Max Welling. "Group Equivariant Convolutional Networks." ICML 2016.\
[3] Hyunsu Kim, et al. "Variational Partial Group Convolutions for Input-Aware Partial Equivariance of Rotations and Color-Shifts." ICML 2024.\
[3] Yulong  Yang, et al. "Learning Color Equivariant Representations." ICLR 2024

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents a neural network architecture intended to be “scale-equivariant and shift-invariant” to photometric corruption, but the equivariance and invariance operate only on global pixel-intensity transformations rather than spatial changes. The method prepends a single convolutional layer with zero-sum weights and uses bias-free layers to enforce these properties, and then applies simple post-processing (e.g., standardization) to obtain affine-invariant predictions. The paper evaluates the approach on CIFAR-10, Cryo-ET classification, and microscopy localization, claiming improved robustness.

### Strengths
1. The paper is reasonably well written and clearly structured.
2. The authors conduct a large number of experiments across several tasks and datasets, providing extensive empirical evaluations.

### Weaknesses
1. The paper initially gives the impression that it addresses spatial affine transformations, but the invariance is only with respect to global intensity changes. Even spatial affine equivariance is already well studied and not particularly novel; restricting the scope further to global intensity scaling and shifting makes the contribution very limited. These transformations are trivial to handle with standard preprocessing or normalization. Removing biases to enforce scale equivariance, or imposing zero-sum weights in the first layer to enforce shift invariance, is not a substantial architectural idea and does not constitute a meaningful contribution.
2. Despite the many experiments, the paper does not seem to have compare against simple preprocessing baselines such as per-image normalization or mean subtraction. These standard techniques may provide the same robustness without any architectural modification. Without such comparisons, it is unclear that the proposed method offers any practical advantage. In fact, enforcing zero-sum filters in the initial layer appears effectively equivalent to preprocessing.
3. Overall, the contribution is very limited. The architectural modification is minimal and conceptually straightforward, and robustness to global intensity transforms can already be achieved with standard pipelines. The work does not demonstrate sufficient novelty or significance.

### Questions
Please see the above comments. My primary concern is the significance, novelty, and timeliness of the work. I would be interested to hear how the authors can clarify the architectural novelty (given that it appears to be a trivial modification) in a way that changes my assessment.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This is a paper which builds on the concept of scale equivariant (SEq) networks, adding shift invariance (SI). This leads to a network which is equivariant to scale operations on pixels in the image, and invariant to shifts in pixel values. In comparison to existing approaches for full affine equivariance, this approach is faster; centrally because it can leverage standard ReLU activations as opposed to the SortPool activations required for full equivariance.

The authors also propose an approach to preserve the logit outputs of their networks through various post-processing stages such as the argmax operation and thresholding.

The authors validate their approach for three tasks including classification and localisation. In all cases, the approach demonstrates strong overall performance combined with robustness both to affine and non-affine corruptions of the images.

### Strengths
The paper is well structured. The mathematics are clearly presented and easy to follow. The experiments are generally well motivated to demonstrate the benefits which the authors claim to achieve.

The results which are presented seem to me to be strong. The authors demonstrate the utility of their approach in a controlled setting (CIFAR-10), a macromolecule classification task and a localisation test. In all cases, the SEqSI framework outperforms performs either competitively or outperforms comparisons without degraded performance in the case of unwarped inputs compared to standard network architectures. 

In particular, experiments where networks are compared with various different transforms at train time, and are evaluated under various shifts at test time, are particularly compelling for this architecture.

### Weaknesses
My central reservation about this paper is the magnitude of its technical contribution. In terms of the technical innovative step, it appears to me that the contribution is principally a zero-sum weight in the first layer of the neural network, and a different transformation of output logits to preserve the equivariance/invariance at output.

Nonetheless, I would support an acceptance of this paper owing to the results which support the view that the method has clear utility. My view would be strengthened if the authors were able demonstrate the effectiveness of their approach more extensively. While the authors do propose several test settings, and the results appear strong, my concerns are as follows:
- CIFAR-10 results appear good, and I would particularly credit the wide range of augmentations used within the evaluation framework. However, this dataset is relatively small and not so complicated, and so these results need to be supported by other results in the paper.
- Macromolecule classification results also appear strong, and this test setting is the most interesting in my view. However, I see that the Affine equivariant and SEq approaches are omitted here, and I cannot find an explanation for this in the paper. I feel that either the inclusion of these baselines, or provision of a reason for their exclusion is essential.
- The 3D localisation test set is also compelling, and the testing framework is strong here. However, as a synthetic dataset it would be useful to have a real-world comparison. This is presented in the form of the Data Science Bowl set (DSB). But in the main paper the DSB is only mentioned as demonstrating the invariance of the approach, whereas I feel that highlighting the robustness of the method to augmentation (Figure 10) was the most important outcome of this test for supporting the method proposed. I feel that if the DSB dataset is mentioned in the main paper, then the paper would be strengthened by including a table or chart of the main results from that dataset in the main paper (i.e. from Figure 10). Alternatively, I feel that demonstrating utility for a different real-world dataset for another task such as segmentation would strengthen the case for this method significantly.

I also found the presentation of the DSB results in the supplementary results to be confusing in some places. In particular I would highlight that the specification that $d=0$ seems to clash with the definition of TP(d) as "the number of pairs of centers *less than* d voxels apart" (emphasis added). I also found the term 'accuracy' in table 16 led me initially to think that this was showing a comparison with ground truth.

### Questions
1. In table 5, your SEqSI network outperforms a standard network both in the no augmentation at train/no augmentation at test, and in the affine at train / affine at test setting. Do you have a good heuristic for why this is the case? Is it because of the network's ability to efficiently learn over variations in the train set? It seems to be a curious result to me if the SEqSI approach is missing both layer biases and has a constraint on the first set of weights. 

2. In figure 10, the selected values of $\mu$ and $\lambda$ are significant in comparison to the range of the original image. As I read it, the chosen value of $\mu$ would mean there was no overlap between the two distributions in the shift case. Is there a reason that such a significant perturbation was chosen?

### Soundness
3

### Presentation
3

### Contribution
2

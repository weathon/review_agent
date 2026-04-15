# Axis-level Reflectional Symmetry Detection with Group-Equivariant Representation

- Decision: Reject
- Scores: 5, 6, 5

## Abstract
Reflectional symmetry detection remains a challenging task in machine perception, particularly in complex real-world scenarios involving noise, occlusions, and distortions. We introduce a novel equivariant approach to axis-level reflectional symmetry detection that effectively leverages dihedral group-equivariant representation to detect symmetry axes as line segments. We propose orientational anchor expansion for fine-grained rotation-equivariant analysis of diverse symmetry patterns across multiple orientations. Additionally, we develop reflectional matching with multi-scale kernels to extract effective cues of reflectional correlations, allowing for robust symmetry detection across different receptive fields. Our approach unifies axis-level detection with reflectional matching while preserving dihedral group equivariance throughout the process. Extensive experiments demonstrate the efficacy of our method while providing more accurate axis-level predictions than existing pixel-level methods in challenging scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
2

### Summary
The paper proposes a group-equivariant neural network for axis-level reflectional symmetry detection。The authors introduce orientational anchor expansion for fine-grained rotational equivariant analysis of different symmetry patterns across multiple orientations. Additionally, the paper develops reflectional matching with multi-scale kernels, enabling robust symmetry detection across various receptive fields. Experimental results demonstrate the effectiveness of the proposed method.

### Strengths
1.	The idea is interesting. Compared to existing methods, which primarily treat reflectional symmetry detection as a pixel-level heatmap prediction problem, this paper classifies the presence of a mid-point of a reflectional symmetry axis for each pixel position and also regress the angle and length of the axis , directly performing axis-level prediction. 

2.	Extensive experiments validate the effectiveness of the proposed method, providing more accurate axis-level predictions than existing pixel-level methods.

3.	The paper is well-organized, making it easy and quick to follow.

### Weaknesses
1.	The literature review is incomplete. The cited references are almost entirely from 2022 and earlier (with only one paper from 2023 and none from 2024), raising questions about the novelty of the work.

2.	The proposed multi-scale expansion has already been widely explored and proven effective in tasks such as object detection and segmentation.

### Questions
1.	In Line 521, you claim that “In Fig. 4, F1-scores for all three methods are plotted across different distance thresholds,” but Fig. 4 only shows two methods, missing the presentation of PMCNet.

2.	The proposed method adapts a line detection network and applies it to the reflectional symmetry detection task. Can the adaptation strategy be effective on other line detection networks?

3.	Could you provide comparison results with existing methods on other datasets (such as SDRW[1] and LDRS[2]) to fully demonstrate the superiority of the proposed method? 

4.	What are the application scenarios, research value, and significance of this study?

[1] Liu, Jingchen, et al. "Symmetry detection from realworld images competition 2013: Summary and results."
[2] Seo, Ahyun, Woohyeon Shim, and Minsu Cho. "Learning to discover reflection symmetry via polar matching convolution."
Flag For Ethics Review: No ethics review needed.

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
5

### Summary
This paper presents a novel axis-level reflectional symmetry detection network that leverages dihedral group-equivariant representations to improve the detection of symmetry axes in images. The authors introduce an orientational anchor expansion method for fine-grained, rotation-equivariant analysis across multiple orientations, enhancing the model's ability to detect diverse symmetry patterns. They also develop a reflectional matching module using multi-scale kernels to capture reflectional correlations across different receptive fields, improving robustness. Extensive experiments demonstrate that the proposed method outperforms existing pixel-level approaches in challenging scenarios, establishing a new benchmark in reflectional symmetry detection. The work offers a fresh perspective and significant contributions to the field of symmetry detection.

### Strengths
* The paper introduces an innovative axis-level reflectional symmetry detection method based on dihedral group-equivariant representations.
* The proposed orientational anchor expansion and reflectional matching modules effectively enhance the model's detection capabilities across various orientations and scales.
* The method demonstrates strong robustness and generalization in complex real-world scenarios.
* The paper provides clear explanations of complex concepts and methodologies, aiding reader comprehension.

### Weaknesses
* The implementation details for orientational anchors could be expanded to clarify their integration within the broader architecture and their impact on computational efficiency.
* While multi-scale reflectional matching is beneficial, further analysis on the trade-off between accuracy and computational overhead would improve the study.
* The model’s applicability to continuous symmetries, such as ellipses or curved patterns, is limited, which may constrain its use in certain symmetry-dense applications.
* The dependency on pre-defined kernels in multi-scale matching might limit adaptability to unknown scales or orientations in real-time applications.
* The paper lacks evaluation of the method's generalization performance on different datasets, which could limit its applicability to other scenarios.

### Questions
* The implementation details for orientational anchors could be expanded to clarify their integration within the broader architecture and their impact on computational efficiency.
* The model’s applicability to continuous symmetries, such as ellipses or curved patterns, is limited, which may constrain its use in certain symmetry-dense applications.
* The paper lacks evaluation of the method's generalization performance on different datasets, which could limit its applicability to other scenarios.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a reflection symmetry detection system with matched, multiscale kernels rotationally equivariant network.  The work uses equivalent networks to allow symmetries to be detected at rotations.   The authors also use a fiber based approach to directly find the symmetry rather than first predicting a heatmap for the symmetry.

### Strengths
The paper is using rotational equivariant networks to improve symmetry detection.  It’s an interesting approach (though needs to be sufficiently distinguished from others in the field). 

This paper is clearly written and I can follow the logic on what they are trying to do.

I appreciate the approach with fibers and it seems interesting to not use a dense approach.    I like the difference in the approach and would like to see more of that with different backbones since I think it would work even better.

### Weaknesses
Major:
For group representation and a longer background of symmetry detection and needs to be cited here is Computational symmetry in computer vision and computer graphics by Yanxi Liu et al.  2010

The evaluation only compares against a recent method and doesn’t go back to any of the previous methods (check out Funk et al 2017 for a list of methods where most are freely available online).  They are used in the papers the authors compared with: Seo, Ahyun, Woohyeon Shim, and Minsu Cho. "Learning to discover reflection symmetry via polar matching convolution." Proceedings of the IEEE/CVF international conference on computer vision. 2021.  In addition, why have you deviated from the standard precision recall with F1 marked curve like the previously mentioned papers?  That is really useful to understand how your metric compares to others.  

This paper (which cited thoughout the paper), also uses equivariant networks for both rotation and reflection symmetry detection.   Seo 2021 and 2022 both use equivariant kernels with Seo 2022 uses rotation group equivalents.  The main difference between the group equivalence of the other papers, at least that I understand, is that the group d8 is used rather than a special Euclidean symmetry group out of the 17.  I'm only referring to the difference in the equivariant and not other differences in the approach.  


Missing citations
Gens, R. and Domingos, P. Deep Symmetry Networks. NeurIPs, 2014.  proposed equivariant convolutional arch that needs to be cited and compared with.  
Rotationally-invariant CCNs - Dieleman, Sander, Kyle W. Willett, and Joni Dambre. "Rotation-invariant convolutional neural networks for galaxy morphology prediction." Monthly notices of the royal astronomical society 450, no. 2 (2015): 1441-1459.

Authors should mention that this is equivariant NN for just 2D data or other papers such as “Equivariant Multi-View Networks. Carlos Esteves et al.  ICCV 2019” should be cited.  

A figure to help understand Sections 3 and 4 would be helpful for understanding what you are getting at here visually.  There is a lot of text and equations and I think a figure to get at the expansion of fibers and how the author sare using symmetry groups would be a big help.


Minor:
First paragraph needs citations.  You can’t just state facts in a paper without a citation on symmetry being a fundamental concept.  You can go back to Gestalt theory or how symmetry detection is prevalent in the animal kingdom but cite it.  

In Figure 1, (a) (b)... needs to be labeled in the image.  This is hard to follow

### Questions
The authors are basing this on Cohen et al.’s work for equivariant cnns and I’m not sure how is this different?  Those filters are already rotational equivariant based on the symmetry groups they represent.  

“Lenc & Vedaldi (2015) show that the AlexNet CNN (Krizhevsky et al., 2012) trained on imagenet spontaneously learns representations that are equivariant to flips, scaling and rotation.” from Group Equivariant Convolutional Network.  Why would this approach be necessary for rotational symmetry invariance for reflection symmetry detection?

How large is the kernel size?  If using too small a size, how can it be 8-fold symmetric?

How well does this objective function work on non-changed neural networks?  What about modern networks like Convnext?

Line 236: why D8 and not some other amount of rotations?

### Soundness
3

### Presentation
2

### Contribution
1

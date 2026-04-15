# Enhancing Robustness of Visual Object Localization by Introducing Retina-Inspired Mapping to Convolutional Neural Networks

- Decision: Withdrawn (Treated as Reject)
- Scores: 1, 3, 3, 3

## Abstract
Foveated vision, a trait shared by many animals including humans, has yet to be fully exploited in machine learning applications despite its key contributions to biological visual function. In this study, we investigate whether retinotopic mapping, a critical component of foveated vision, can improve image categorization and localization performance when incorporated into deep convolutional neural networks (CNNs). In particular, we incorporated log-polar retinotopic mapping into the inputs of classic off-the-shelf CNNs and retrained these network on the ImageNet task. Surprisingly, the retinotopically mapped network performed equally well in classification but showed improved robustness to arbitrary image zooms and rotations, especially for isolated objects. In addition, this network showed improved classification localization when the foveated center of the transform was moved, mimicking a key capability of the human visual system that is lacking in standard CNNs. These results suggest that retinotopic mapping may underlie important preattentive visual processes.

## Human Reviews

## Human Reviewer 1

### Rating
1: strong reject

### Rating Number
1

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors use a log-polar representation of images to train rotation- and scale-invariant image classifiers. They find that the log-polar versions of VGG-16 and ResNet-101 are more robust to image rotations than their (standard) Cartesian versions. They also find that placing the center of the log-polar representation at different places in the image can be used for object localization.

### Strengths
+ More human-like vision systems are an interesting and potentially important research direction to improve robustness
 + Well-written, straightforward to follow

### Weaknesses
### 1. Lack of novelty

The authors' approach is basically a Polar Transformer Network (Esteves et al., ICLR 2018). Unlike the original work (which they don't cite) they don't even address the question of how to determine the polar original, but just take labels from the ImageNet dataset. 


### 2. Robustness is trivial

The authors motivate their work by adversarial robustness, but the actual approach doesn't get there at all. Instead, they test robustness against rotation, which is trivially better in a Polar Transformer Network because it is equivariant to rotation and scale by construction. Thus, I don't see how this approach brings us any closer to robustness in deep networks.

### Questions
None.

### Soundness
2 fair

### Presentation
3 good

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
The authors investigate the potential of foveated vision—a feature in biological systems—in improving the performance of off-the-shelf CNNs. This is achieved by applying a log-polar transformation to the input images and retraining the CNNs. While the model trained with retinotopic inputs demonstrated similar classification performance compared to its standard counterpart trained with Cartesian images, it showcased superior robustness to image zooms and rotations, as well as improved classification localization.

### Strengths
- The writing is well-structured and easy to follow.
- The method is clean and solid.
- The results on visual object recognition are inspiring and may suggest the proposed approach can be applied to a dynamic visual model that incorporates the human saccadic eye moments.

### Weaknesses
- The novelty of this work is unclear to me, especially given the prior existence of similar concepts, such as Polar CNNs [1]. There are also many prior works applying foveated images to deep neural networks, and evaluated their robustness to adversarial attacks [2,3]. These related works are not discussed in the paper.
- The robustness on rotated images is expected given that the CNNs trained with polar-transform images inherently extract rotation-invariant features. Besides, this idea was also already validated in [1].
- The visual object localization experiments, while insightful, are not surprising and may result from an "unfair" comparison. Given the non-uniform sampling of the retinotopic input, it naturally narrows the "effective" FOV when you use the exact same 8x8 grid to compare Cartesian input vs. Retinotopic input.

### Questions
- Why rotational invariance is considered as a biologically plausible property? Previous studies [4] suggest that visual recognition in humans depends on the viewing angle. Also, the retinotopic mapping in the human visual system does not lead to an inherent invariance to image rotations.
- When retraining models such as VGG116 and ResNet101 with log-polar transformed input, did you apply circular padding for the $\theta$ axis?
- Did the kernels in CNNs retrained by log-polar transformed images also manifest meaningful feature extractors (e.g., edge detection in lower layers and shape recognition in higher layers)? 

minor points: It's unconventional (and incorrect in my opinion) to refer to rotated images as an "attack".

**Referebce**:

1. Esteves, C., Allen-Blanchette, C., Zhou, X., & Daniilidis, K. (2017). Polar transformer networks. ICLR 2018.
2. Luo, Y., Boix, X., Roig, G., Poggio, T., & Zhao, Q. (2015). Foveation-based mechanisms alleviate adversarial examples. ICLR 2016.
3. Vuyyuru, M. R., Banburski, A., Pant, N., & Poggio, T. (2020). Biologically inspired mechanisms for adversarial robustness. Advances in Neural Information Processing Systems, 33, 2135-2146.
4. Lawson, R. (1999). Achieving visual object constancy across plane rotation and depth rotation. Acta psychologica, 102(2-3), 221-245.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces the idea that having a log-polar transformation at the start of a CNN (as a pre-processing stage) improves the general robustness/accuracy of an animal localization task, while maintaining classification performance. The Authors also are interested in shedding light on the question of what are the computational advantages (if any) of foveation.

### Strengths
* This paper's main strengths are the scientific questions authors are trying to address. However, I am not sure that the logic is fully correct and the experiment prove their main point. The question however is very interesting, and not addressed enough in the field: Why do humans foveate, and what advantages can machines get of such spatially-adaptive computation? This it the type of questions that are very hot in the emergent field of NeuroAI.
* The authors use other datasets other then ImageNet (they use Animal-10k) -- though see Weaknesses, I am not sure if this is a good decision.
* The notion of adding scale and rotational invariance as a pre-processing transform is interesting, but I also think that this has been addresses in other ways with multi-scale transformer archtiectures such as CrossViT (Chen et al. 2021)

### Weaknesses
* Not enough experiments. Also not sure how the brittleness of lack of rotational invariance (Figure 1) is later addressed in Table 1. Unless Figure 1 already shows the final result with the color-tone but after re-reading the caption, I don't think this is the case.
* Weird experimental selection : Why Animal-10k dataset over something like ImageNet (Objects) or Places (Scenes)? 

--------

There are a lot of **relevant missing papers** regarding the question of "What is the purpose of Foveation?" and similar in the previous work section (and that can contribute to the discussion). See below:

Key Missing Critical References:
- Deza & Konkle. ArXiv, 2021. Emergent Properties of Foveated Perceptual Systems.
- Wang & Cottrell. Journal of Vision, 2017. Central and peripheral vision for scene recognition: A neurocomputational modeling exploration.
- Cheung, Weiss & Olshausen. ICLR 2017. Emergence of foveal image sampling from learning to attend in visual scenes

Secondary, but also important References:
- Gant, Banburski & Deza. SVRHM, 2022. Evaluating the adversarial robustness of a foveated texture transform module in a CNN.
- Reddy, Banburski, Pant & Poggio. NeurIPS 2020. Biologically inspired mechanisms for adversarial robustness
- Wang, Mayo, Deza, Barbu & Conwell. SVRHM, 2021. On the use of Cortical Magnification and Saccades as Biological Proxies for Data Augmentation
- Harrington & Deza. ICLR, 2022. Finding Biological Plausibility for Adversarially Robust Features via Metameric Tasks

### Questions
- Table 1: Shouldn't the probably of fixation in animal and out animal sum to 1 in the aggregate? Or not necessarily?
- Figure 5 : What is accuracy here? Is it a percentage or a ratio? The top value of $10^{-2}$ would mean that the highest accuracy is 1% (0.01), or is this a typo? Should it be $10^2$ instead?
- Figure 6 (Supplement) looks strange. Why is the log-polar transform being computed locally per each small region, vs over the whole image given a point of fixation.
- Given the previous question. What is the point of Figure 6 given Figure 2 -- which seems like what the Authors are doing.

I am open to changing my score, perhaps I did not understand the authors key contributions, and they are welcome to address many of my concerns in their rebuttal. 

I am not rejecting the paper due to lack of innovation (novelty) [The paper poses a really interesting question, and approach], but rather because I am not fully convinced or understand what the authors intend to show in the paper through their limited experiments -- including those in the Supplementary Material.

### Soundness
3 good

### Presentation
2 fair

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
The authors propose the incorporation of foveated visual processing into deep convolutional neural networks (CNNs). The authors pre-process images with a log-poral mapping before passing the images through off-the-shelf CNNs for (re)-training and evaluation. The authors show that the incorpoated foveated processing improves the robustness to scale and rotation perturbations while retaining classification accuracy on non-perturbed inputs. The authors also show that the foveated network produces improved classification localization when the fovea center was moved, which is not possible to perform in standard non-foveated CNNs.

### Strengths
+ This submission proposes a simple extension of pretrained off-the-shelf CNNs with a log-polar retinotopic transform which enhances the rotation and scale invariance of the learned representations while retaining accuracy on non-perturbed upright images.
+ There are interesting discussion sections on connecting the proposed architecture with pre-attentive mechanisms and eye movements in visual processing.
+ The writing in this paper is very clear and the visualizations (esp. Fig 2 illustrating log-polar transforms) are helpful to improve the readability of the submission. In my opinion, the paper is quite easily accessible for both computer vision and neuroscience audience.

### Weaknesses
- The proposed approach lacks novelty; contrary to the authors claiming to introduce the biologically-inspired log-polar retinotopic mapping to CNN inputs, this has been explored in the past [1, 2, 3], the authors haven't added this related work in their paper and state that log-polar transforms in CNNs are largely underutilized. [2] explores object localization performance on rotated images which is one of the core premises of the current submission.
- The proposed work uses VGG-16 and ResNet-101 networks which are far behind the current state of the art in computer vision. This reduces the impact of the proposed work for machine learning. I would suggest the authors to please use more recent architectures with higher classification performance (on both clean images and rotated images) if they would like to make a strong contribution towards rotation invariant neural networks.
- While using off-the-shelf networks to show improved rotation and scale invariance, the authors are restricted from reporting how significant the observed gains are over multiple random seeds. I would suggest adding both stronger baselines (in relation to my previous point) and evaluating performance over multiple random initializations in order to provide a fuller picture of how important log-polar mapping is to rotation invariance. If the authors are to make this submission more exciting to the computer vision community, they must clearly state how such input transformations are helpful (and feasible) in a time where pre-training full models is less viable as the industry moves towards finetuning task-specific decoders on top of strong frozen backbones (which seem to possess strong generalization abilities already). 
- Overall, I find this submission to not be making very exciting contributions to either computer vision or neuroscience communities and that it could be significantly improved before publication at ICLR.

References:
1. Remmelzwaal, L. A., Mishra, A. K., & Ellis, G. F. (2020, January). Human eye inspired log-polar pre-processing for neural networks. In 2020 International SAUPEC/RobMech/PRASA Conference (pp. 1-6). IEEE.
2. Cao, J., Bao, C., Hao, Q., Cheng, Y., & Chen, C. (2021). LPNet: Retina inspired neural network for object detection and recognition. Electronics, 10(22), 2883.
3. Gahl, M., Kulkarni, S., Pathak, N., Russell, A., & Cottrell, G. W. (2022). Visual Expertise and the Log-Polar Transform Explain Image Inversion Effects.

### Questions
Please refer to my weaknesses section above.

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

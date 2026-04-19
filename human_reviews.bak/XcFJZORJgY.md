# Discrimination for Generation

- Decision: Reject
- Scores: 8, 5, 5

## Abstract
There are two primary approaches to learning from data: discriminative models, which make predictions based on provided data, and generative models, which learn data distributions to create new instances. This paper introduces a novel framework, Discrimination for Generation (DFG), as the first attempt to bridge the gap between discriminative and generative models. Through DFG, discriminative models can function as generative models. We leverage the Neural Tangent Kernel (NTK) to map discriminative models into a connected functional space, enabling the calculation of the distance between the data manifold and a sampled data point.
Our experimental results demonstrate that the proposed algorithm can generate high-fidelity images and can be applied to various tasks such as Targeted Editing and Inpainting, in addition to both unconditional and conditional image generation.
This connection provides a novel perspective for interpreting models. Moreover, our method is algorithm-, architecture-, and dataset-agnostic, offering flexibility and proving to be a robust technique across a wide range of scenarios.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The method repurposes discriminative models as generative models by utilizing the Neural Tangent Kernel (NTK) based on the gradient of pretrained model as a function to implicitly map discriminative features into a connected space of data manifold, without additional training. This demonstrates that nearly all discriminative methods can be transformed into generative models regardless of architecture, dataset, or algorithm, though the initial results may not be as performant as later ones. With NTK technique, the paper also present a new technique to visualize features by using model parameter itself.

### Strengths
- The method is theoretically sound, offering valuable insights in relation to score-matching frameworks, which also implicitly aim to minimize the distance between generated samples and the difficult-to-approximate data manifold.
- The idea of using Neural Tangent Kernel to implicitly map features into data manifolds is an elegant and substantial contribution.

### Weaknesses
- Although the method has theoretical guarantees, the output images still lack structural richness and differ significantly from natural input image.
- In DFG part at L347, the intuition of using augmentation-invariance to emphasize the importance of each class is unclear. It is confusing because what is the relationship between importance of image class and the difference of features between original inputs and its augmented versions. Justification or empirical evidence is needed.
- Does the assumption of a corresponding minimization between Functors and their inputs always hold? (Eq. 5) The authors should provide theoretical justification or empirical evidence for this assumption. Meanwhile, is there any edge case that breaks this assumption?
- The annotation in Figure 3 is unclear; I’m confused about whether the upper row of 6 figures represents a zoomed-in version of each variable  $x_t$  below or conveys a different meaning at a specific time step. A more detailed caption is needed to describe the chart. 
- What is the choice of distance function $d(.)$
- Is the method applicable to the representation of generative models? If so, it would be interesting to see if the proposed approach could transform an unconditional diffusion model into a conditional diffusion model using discriminative models, without additional training. As in section 5.2, the authors have shown that method is extended to conditional generation in Eq 13 with score term and condition term. In this case, the score term can be instantiated with score-based model.
- The applications part is very interesting but the quality is not that impressive. The outputs exhibit noticeable artifacts and unnatural frequency patterns.
- It is recommended to include quantitative metrics on image generation to enhance the quality of the paper (e.g. FID, IS, classification accuracy for class-conditional generation).

### Questions
Please address my concerns above! I highly value this paper for providing new insights into the representation space between generative and discriminative models, even though the results are not as compelling as those of generative counterparts. The findings can pave the way for next advancements in the generative era.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper proposes to generate a data sample by aligning its NTK feature with the average NTK feature of a real dataset. The authors approximate the latter with the parameters of the trained network (e.g., for classification). Therefore, given a trained model, the algorithm iteratively updates a synthetic sample s.t. its induced Jacobian (i.e., the NTK feature wrt this trained model) is similar to the model parameters, where the output dimension is collapsed via a weighted average. The authors experiment with unconditional and conditional generation, impainting, editing, and data understanding.

---
I have interacted with the authors, read their response, and decide to keep my original evaluation. In particular, the authors did address some of my concerns, but relied on large changes to the submitted manuscript. Some other concerns, e.g., if generation via discriminative nets are even possible, are only discussed with speculation and thus not fully addressed. And new issues, such as extremely unfair comparison (e.g., Fig 11) emerge during the discussion period. I recommend the authors develop the project further and it could become a strong submission at another venue.

### Strengths
+ Novel exploration for using NTK alignment in discriminative for generation.
+ Diverse experiment setting with nice visualizations.

### Weaknesses
+ The approximation of the average real NTK feature with trained parameters is not well explained. The text points to Radhakrishnan et al., 2023, but I fail to find a justification in that paper. 
+ The DFG dynamic weighting is not well-analyzed. Practically, what's the performance without it? Theoretically, what stationary solutions could it converge to?
+ Missing discussion on obvious generation artifacts. Since the paper claims to use discriminative networks for generation (instead of, say, purely dataset understanding), it is important to discuss whether these are limitations of the method or the idea of using discriminative networks alone.
+ Unclear how the proposed method performs "Global Explanation" or "feature visualization" (section 6.3) better than compared baselines. Should we consider the generated image as one that captures important features picked up by the discriminative network? If so, why is realism claimed as an important advantage?
+ Missing qualitative generation evaluations such as FID, human evaluation, etc.
+ Missing discussion with Dataset Distillation works [1], some of which also generate samples by aligning deep features [2], aligning gradients [3], or requiring synthetic data to induce the same classifier network [4]. These works also generate interesting-looking images that can be used for dataset understanding.


[1] Dataset Distillation. Tongzhou Wang, Jun-Yan Zhu, Antonio Torralba, Alexei A Efros. arXiv 2018
[2] Dataset Condensation with Distribution Matching. Bo Zhao, Hakan Bilen. WACV 2023.
[3] Dataset Condensation with Gradient Matching . Bo Zhao, Konda Reddy Mopuri, Hakan Bilen. ICLR 2021.
[4] Dataset Distillation by Matching Training Trajectories. George Cazenavette, Tongzhou Wang, Antonio Torralba, Alexei A Efros, Jun-Yan Zhu. CVPR 2022.

### Questions
+ See above for weaknesses.
+ What if the method uses some other feature maps than NTK?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper proposes a generative approach that uses pre-trained discriminative models (such as DETR, DINO, ResNet) for image generation. The proposed approach learns to generate samples by calculating the distance between the data distribution and sampled point using a kernel inspired by Neural tangent Kernel and shows similarity to score function used in diffusion models. The paper proposes Discrimination for Generation ( DFG) approach that could be used in conditional and unconditional generation setting. The paper also shows analysis of how the proposed approach is equivalent to approximating score function in diffusion models. The paper shows qualitative results on generative/in-painting tasks to demonstrate the efficacy of the proposed approach.

### Strengths
The paper explores an approach that transforms a pre-trained discriminative model (in domains of image segmentation, detection, classification) into generative models. The approach explains how we can use an approximated kernel to compute the distance between sampled data points and data distribution. The proposed approach shows results on different types of discriminative models for conditional as well as unconditional generation. The paper provides theoretical analysis to explain the proposed distance metric. The paper is easy to follow and shows results on in-painting and explanation tasks.

### Weaknesses
It would be helpful for a reader to get a better understanding of the following:

1. It would helpful for the reader to see some discussions around the time complexity of each inference call and some comparison with other baselines. It would be great to see specific runtime comparisons and complexity analysis with baselines such as SOTA diffusion models(eg. SDv2) and GANs (eg. StyleGANs)

2. Quantitative evaluation metrics (FID, IS) on standard datasets (e.g. CIFAR-10, ImageNet) with other generative baselines (diffusion models - stablediffusion v2.1 or XL, GANs - styleGANv2) could help the readers in evaluating the efficacy of the approach as the generated sample don’t look like high quality as mentioned in the paper.


3. It would helpful for the reader to see some discussions around Energy based models and even other approaches that leverage discriminative models for image generation. For example, discussion around referred energy-based [1] or discriminative generation methods [2] and how DFG approach compares or differs.

 It would be helpful for the readers if some more related generative papers are added as references:
 [1] Duvenaud, David, et al. "Your classifier is secretly an energy based model and you should treat it like one." ICLR 2020 
[2] Santurkar, Shibani, et al. "Image synthesis with a single (robust) classifier." Advances in Neural Information Processing Systems 32 (2019)

Minor typo

line 103 similarly -> similar

### Questions
Should there be a negative sign in equation 9 instead of addition sign?

### Soundness
3

### Presentation
2

### Contribution
2

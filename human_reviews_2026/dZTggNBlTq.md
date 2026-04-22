# 1S-DAug: One-Shot Data Augmentation for Robust Few-Shot Generalization

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 6

## Abstract
Few-shot learning (FSL) demands generalization to novel classes based on just a few shots of labeled examples, a setting where traditional test-time augmentations fail to be effective. We introduce 1S-DAug, a one-shot generative augmentation operator that synthesizes diverse yet faithful variants from just one example image at test time. 1S-DAug couples traditional geometric perturbations with controlled noise injection and a denoising diffusion process conditioned on the original image. The generated images are then encoded and aggregated, alongside the original image, into a combined representation for more robust FSL predictions. Integrated as a training-free model-agnostic plugin, 1S-DAug consistently improves FSL across standard benchmarks of 4 different datasets without any model parameter update, including achieving 10% proportional accuracy improvement on the miniImagenet 5-way-1-shot benchmark. Codes will be released.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors introduce a new method to create additional samples which are different yet close enough to support few-shot image classification, especially in the lowest 1-shot setting. The approach manages to introduce variation while staying true to the core class information via a combination of geometric transformation, noise injection and targeted denoising via a ‘true’-sample-conditioned diffusion model.

### Strengths
**Originality & Significance:**  
- The authors do a great job in outlining why the task matters, as well as how it is approached; Especially the introduced angle of looking at the value of data augmentation from the angle of ensembles adds a good layer of depth to ease the reader into the later presented method
- The idea is easy to understand, and can be employed in a model-agnostic manner (given it’s only modifying/adding to the data)
- For people with knowledge in FSL as well as diffusion models, it is also a seemingly 'obvious' way that 'should' work, and the authors do a good job of clearly outlining the ingredients that are required to make it work in practice

**Quality:**  
- The work is placed well within related efforts, and the specific gap the authors tackle is clearly presented and justified

**Clarity:**
- The work is mostly well written and easy to follow, and a good mix of illustrations (esp. Figure 2) and text makes the core concepts and effect of the augmentations easy to grasp

### Weaknesses
- **Parts of the few-shot ‘basics’/background are misrepresented** in the current manuscript, e.g. l.166 ff: *’For inductive FSL, model parameters are frozen during inference, while the transductive set-up permits test-time adaptation.’*  
$\rightarrow$ This is *not* the common definition of inductive vs. transductive, which is in contrast usually defined via the access to the unlabelled test-time samples: inductive models can very-well be finetuned on the support set (e.g. see seminal MAML) but do not (yet) have access to the unlabelled query set; whereas the union of support and query set is available from the start in transductive settings (i.e. the unlabelled information can be exploited as well)!
- **’Seminal’ choice of models**: The choice of models is limited to a toy 4-layer ConvNet and a 12-layer ResNet; Although these have been popular in the field for a long time, most recent works have demonstrated results with ViT-/Swin-variants (single or multi-scale). Given that the authors’ contribution is on the data-augmentation side and the fact that Transformer-based models do behave differently, results in this space would significantly strengthen the paper.
- **Only in-domain experiments**: Especially since the work presents a data-augmentation approach, I’d be highly interested to see how the findings transfer to cross-domain experiments (which are common in FSL)
- **Very ‘selected’, partially outdated and somewhat misleading SOTA comparison**: Newest compared method is from 2023 (Table 1); Note that newer works that use ResNet12 exist, and even some older ones perform better than the reported methods (e.g. PAL, ICCV 2021; COSOC, NeurIPS 2021, etc.);  As mentioned previously, Transformer methods would additionally strengthen the paper (e.g. FewTURE, NeurIPS 2022; HCTransformers, CVPR 2022)
- Lack in clarity why some results not reported in Table 1, as well as limited discussion of reported ones; see questions.
- Some lack in clarity around interpretation of the insights in Table 4; see questions.

### Questions
**TL;DR:** While I think the paper presents an interesting and straight-forward approach that could have quite a wide range of applications across FSL methods, I’m missing a number of key insights at this stage (see weaknesses above and questions below); If the authors can address (some of) these, I’m more than happy to reconsider my current rating. 

For more details, please also see the weaknesses above.   
My main questions relating to these:  
- Can the authors provide insights into the impact of their methods on recent Transformer backbones?
- Can the authors provide insights and/or background information how their method would perform in cross-domain settings? I think this would significantly strengthen the paper, given that the domain gap usually challenges models, and it would be of interest whether augmentation in this manner could improve robustness & performance. (E.g. miniImagenet $\rightarrow$ CUB, which should be easy to evaluate; even more interesting would be the meta-dataset, but likely not feasible given limited time)  
- Re Table 1: I don’t quite understand why results for some 1S-DAug methods are not reported, e.g. 1S-DAug-3 for tiered 1shot, etc?
- Re Table 4: The results show that augmenting the query as well is almost required to obtain ‘good’ results; However, doesn’t this actually challenge the authors underlying assumption of ‘better representing the true underlying distribution’ of the data where the support set is sampled from?   
  How exactly are you computing the accuracy when you have multiple queries – are the query embeddings averaged (kind of query-prototypes), or are they individually classified and you aggregate the results (e.g. voting, etc.)?   
  $\rightarrow$ If averaging, do you have insights as to why this helps to match better? I assume dimensions of high variation would be attenuated, and the mean moves closer to the prototype mean; but in theory, shouldn’t having more support samples be able to achieve the same goal (i.e. move the mean closer to the ‘true’ centre and therefore closer to the query)? Or does the diffusion process introduce new information/create patterns that makes matching simply easier? (Or is there simply something I’m missing?)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces 1S-DAug, a training-free, test-time, one-shot generative augmentation operator for few-shot classification. Given a single image, 1S-DAug (i) applies class-preserving geometric tweaks, (ii) injects a controlled amount of diffusion noise, and (iii) performs image-conditioned denoising to synthesize a handful of diverse yet class-faithful variants. Features from the original and generated views are aggregated (simple averaging) and fed to standard metric-based few-shot heads (e.g., ProtoNet/FEAT), without updating any model parameters. On four benchmarks (miniImageNet, tieredImageNet, CUB, Animals), the method yields consistent gains over the corresponding non-augmented baselines; the paper also provides an intuitive risk-decomposition showing why ensembling the augmented view can reduce error when accuracy is preserved and predictions are decorrelated, plus a margin/complexity discussion in the appendix.

### Strengths
- The paper is clearly written and the pipeline is easy to follow.
- Works at test time with standard few-shot backbones (ProtoNet/FEAT), requiring no fine-tuning of the classifier or the generator. This makes it practical when model parameters are fixed or restricted. 
- Some theoretical backing. The risk decomposition (Eq. 12) formalizes the accuracy-vs-diversity trade-off for two-view ensembling. While simplified, this helps motivate the design.

### Weaknesses
1. The method is quite simple. It mainly combines basic geometric perturbations with diffusion-based denoising and feature averaging. While this simplicity is a strength in terms of clarity, it also limits novelty. In addition, the evaluation mostly compares against older baselines such as ProtoNet and FEAT, without including more recent few-shot generation or subject-level adaptation methods, such as [1, 2] and also Fsl-rectifier. These newer works would make the comparison fairer and show whether the proposed approach can still offer advantages.  
2. According to Table 5, the overall improvement seems to rely heavily on the Generation Techniques. This suggests that most of the gains come from the generation process itself, while the other components contribute only marginally.  
3. The method depends on a pretrained Stable Diffusion 1.5 model, which already encodes rich visual priors for most classes. It’s unclear how the approach would perform for categories that are truly novel to the generator. In such cases, the results might not be better than simple data augmentation. Conversely, with stronger modern generators (e.g., Flux or newer SD variants), the method might perform better. It would be useful to discuss how sensitive the method is to the underlying generator’s prior knowledge.

[1] ProtoAug: Provably Improving Generalization of Few-Shot Models with Synthetic Data

[2] DataDream: Few-shot Guided Dataset Generation

### Questions
Please refer to Weaknesses.

### Soundness
2

### Presentation
3

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
The paper introduces 1S-DAug, a method for performing few shot learning by taking the existing support set, applying geometric transforms for increasing coverage of layout and pose, and add controlled noise that distorts high-frequency content like details and artifacts while preserving the content of the images, and finally using a denoising using a diffusion model to generate augmented support samples. The model does not use the labels in any way to generate augmented samples, potentially avoiding the need for the test samples to have any similarities or distributional overlap to the training datasets.

### Strengths
1. The paper proposes a technically sound and exciting paradigm to few-shot learning by introducing additional support set examples at inference, instead of attempting to make the FSL network more robust during training. Test-time augmentation has been successful in many other computer vision and image processing tasks including segmentation, classification, depth prediction, etc., and this paper adds to the relatively scarce literature on essentially what constitutes test-time augmentation in FSL. The method can be applied to most base FSL methods without additional retraining, making it a strong plug-and-play method for variety of few-shot learners.
2. The experiment setup is strong, with relevant baselines and settings (5-way 1/5-shot learning). The improvements are consistent across datasets and few shot configurations.

### Weaknesses
1. **More baselines, comparison, and discussion**: FSL-Rectifier is a generative baseline to perform data augmentation during test-time inference. however, other methods perform use adversarial methods like adversarial GAN to generate harder examples [1], or adversarial geometric distortions [2]. Although these methods use the adversarial component during training, they can be used at inference, and their adversarial training could make performance more robust. Adding these methods to comparisons and discussions might further differentiate why the proposed method is better than or does not require sophisticated adversarial objectives for few-shot learning. The differences in augmentations considered in the paper versus those used in [2] and its implications should also be included in discussion (particularly, if [2] can be used to train an adversarial training module instead of sampling generations to produce a sharper decision boundary [1]). Another potential line of work that can add to the discussion is the role of few-shot generation methods [3] that are possibly less biased than a pretrained diffusion model on a fixed set of classes that may not be seen during inference. A very brief discussion is provided in the end of Section 6, but that could be expanded upon in the main paper or appendix. Figure 4 and "More Related Work" is a good starting point.
2. Theoretical justification is a little weak in general - Section E2 assumes that the test and augmentation sets are drawn iid from the same distribution - which is too strong of an assumption to make. An estimate of distributional distance between the test and augmentation sets and classifier error bounds being a function of this distance would be a more meaningful result to give the user an idea of how well their classifier might perform given a diffusion model and its capacity to represent the test distribution accurately. Equating the distributions just derives the effect of adding more samples into the classifier, which will lead to better decision boundaries - it is a known result, and fairly straightforward to derive. 
3. Section 3 introduces the preliminaries for a reader who is already familiar with diffusion models. I would recommend introducing only the minimal notation necessary for understanding the main paper and redirecting the reader to a more detailed supplementary section or external paper for details on the DDPM/DDIM objective/learning/noise scheduling/sampling, etc. In its current form, Section 3 comes across as very hasty and does not add much value to the paper in my opinion. 
4. Most diffusion models in the literature are text-conditioned, but not image conditioned, which limits the applicability of the method from existing pretrained models like StableDiffusion. The model must evaluate on text-conditioned or unconditioned diffusion models (i.e. provide ablations on the diffusion model) to see if the few-shot performance is preserved or degraded by the diffusion. 
5. The section on theoretical justification is hand-wavy at best, and at best provides sufficient conditions for the risk of the augmented classifer being less than the risk of the base classifier, which are trivial in some sense. Specifically, if $\mathbb{E}[f_{\mathcal{A}}(x)l] \ge \mathbb{E}[f(x)l]$ it already means that the classifier with the augmented samples performs better than the base sample, and in that case the base sample can even be removed from the FSL. The second diversity term is always non-positive since $f$ is constrained to be $\le 1$, making only the first term an indicator of whether the risk reduces by augmentation, which is a self-fulfilling condition. A more meaningful comparison could be if the risk can somehow be proxied with some property or statistic of the augmented samples relative to the input sample which is non-trivial. I urge the authors to reduce the claims in Section 5 (since the condition for risk-reduction reduces to $\mathbb{E}[f_{\mathcal{A}}(x)l] \ge \mathbb{E}[f(x)l]$ which is better label agreement of the augmented samples - a trivial and self-fulfilling condition that also cannot be checked at inference time). 
6. The reimplementations consistently score _lower_ than the numbers mentioned in the paper. Comparing with the metrics reported in the baselines immediately reduce the accuracy gap by $\sim 2\%$, weakening the claims of performance improvement of the method. Is there any specific reason why existing methods were not used? 
7. The purpose of Table 3 is unclear - what is the purpose? It is known that diffusion models will take more inference steps (longer runtime). Section 6.2 can be moved to the appendix without disrupting the main claims and narrative of the paper. This section can be replaced with a plot of noise level versus performance for all the datasets and fewshot configurations and a discussion on whether the performance is dependent on the chosen $t_0$.

Minor nits:
1. Line 375 should refer to Table 4 and not Table 5.
2. Line 782 - "not suitable" -> "that are not suitable"
3. What are the shape tweaks used in the paper? I could not find a description in the main or supplementary paper.

[1] https://proceedings.neurips.cc/paper_files/paper/2018/file/4e4e53aa080247bc31d0eb4e7aeb07a0-Paper.pdf 

[2] https://openaccess.thecvf.com/content_CVPRW_2020/papers/w54/Jena_MA3_Model_Agnostic_Adversarial_Augmentation_for_Few_Shot_Learning_CVPRW_2020_paper.pdf 

[3] https://arxiv.org/abs/2205.15463

### Questions
1. In Line 101, the paper mentions "Since our augmentation is strictly one-shot during inference, it can fulfill the challenging FSL test-time augmentation requirement." -- It is not clear what the "strictly one-shot" portion is. What is a FSL test-time augmentation requirement? 
2. The image-conditioned attention provides a shortcut to the denoiser - it can essentially ignore the value of $z_t$ and simply recover the image from its conditioning. How is this mechanism prevented in learning the diffusion model? 
3. What is the diffusion model used in the paper? Please mention one sentence in the Evaluation Protocol, or point the reader to the excerpt in the paper where this is mentioned. The diffusion model seems to preserve the object identity well compared to a model without image conditioning (Figure 2). The details of the diffusion model are therefore important to understand the mechanisms of original-image conditioned image generation.

### Soundness
3

### Presentation
3

### Contribution
3

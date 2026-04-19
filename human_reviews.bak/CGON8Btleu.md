# BrainACTIV: Identifying visuo-semantic properties driving cortical selectivity using diffusion-based image manipulation

- Decision: Accept (Poster)
- Scores: 6, 8, 8, 5, 6

## Abstract
The human brain efficiently represents visual inputs through specialized neural populations that selectively respond to specific categories. Advancements in generative modeling have enabled data-driven discovery of neural selectivity using brain-optimized image synthesis. However, current methods independently generate one sample at a time, without enforcing structural constraints on the generations; thus, these individual images have no explicit point of comparison, making it hard to discern which image features drive neural response selectivity. To address this issue, we introduce Brain Activation Control Through Image Variation (BrainACTIV), a method for manipulating a reference image to enhance or decrease activity in a target cortical region using pretrained diffusion models. Starting from a reference image allows for fine-grained and reliable offline identification of optimal visuo-semantic properties, as well as producing controlled stimuli for novel neuroimaging studies. We show that our manipulations effectively modulate predicted fMRI responses and agree with hypothesized preferred categories in established regions of interest, while remaining structurally close to the reference image.  Moreover, we demonstrate how our method accentuates differences between brain regions that are selective to the same category, and how it could be used to explore neural representation of brain regions with unknown selectivities. Hence, BrainACTIV holds the potential to formulate robust hypotheses about brain representation and to facilitate the production of naturalistic stimuli for neuroscientific experiments.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors introduce BrainACTIV, a method for generating images modulated by the responses of a specific brain region, enabling interpolation between maximum and minimum activation levels. Additionally, they present two hyperparameters that regulate the semantic and structural variations of the generated images. The authors claim that this approach has the potential to provide insights into neuroscientific experiments.

### Strengths
1. The authors first achieved controllable image generation by manipulating a reference image to enhance or suppress specific target brain regions.
2. Based on the example images, particularly the interpolated manipulated reference images in Figure 1, the proposed method appears effective, especially concerning the functional regions of interest defined by neuroscience.
3. The article is highly readable, thanks to its clear writing and presentation.

### Weaknesses
1. The motivation needs to be better articulated. While the article emphasizes the controllable modification of the reference image, it lacks a clear rationale for this approach. What advantages does this controllable generation offer for guiding neuroscience discoveries compared to existing methods?
2. The technical novelty is limited, as the article primarily implements IP-Adapter and SDEdit to achieve controllable modifications of reference images.

### Questions
1. A comparison with related work [1-2] would be valuable if feasible.
2. The response to brain activation seems to be influenced by the fMRI encoder (specifically the CLIP encoder used in this paper), so its performance should be addressed in the experimental section.
3. How do the four mid-level features—entropy, metric depth, Gaussian curvature, and surface normals—contribute to advancements in neuroscience discoveries?
4. How do the two hyperparameters regulate the trade-off between semantics and structure? Increasing either hyperparameter seems to lead to deviations from the reference image, but it’s unclear how this establishes a trade-off between the two aspects. What is the rationale behind this approach?

[1] Gu Z, Jamison K W, Khosla M, et al. Neurogen: activation optimized image synthesis for discovery neuroscience[J]. NeuroImage, 2022, 247: 118812.

[2] Luo A, Henderson M, Wehbe L, et al. Brain diffusion for visual exploration: Cortical discovery using large scale generative models[J]. Advances in Neural Information Processing Systems, 2024, 36.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes a method for manipulating a reference image so as to selectively enhance (or decrease) the corresponding predicted brain activity in a given target region. The image manipulation is performed with pretrained diffusion models, and the brain activity prediction relies on training a brain decoder to perform regression between a large image dataset (NSD) and the corresponding brain activation patterns.

### Strengths
* The proposed image manipulation technique is non-trivial, and goes much beyond deploying an off-the-shelf diffusion system. It leverages state-of-the-art diffusion systems and adapters.
* The manipulated images are convincing, and compatible with known selectivity in the target brain regions. 
* The technique allows to draw further subtle distinctions between brain regions that have been traditionally associated with similar category selectivity, and could be used to inform future neuroscience experiments.
* The method allows to control the trade-off between semantic variation and structural similarity with two hyperparameters.

### Weaknesses
The paper shares weaknesses with most brain decoding studies, in the sense that the results describe predictions of a model trained from a specific dataset and a specific (combination of) deep learning model(s).  These predictions will ultimately require validation from actual experiments. However, this is well acknowledged in a paragraph on “limitations”.

### Questions
* “we opt to average the embeddings over all subjects, excluding the subject on which predictions are made. Hence, we modulate brain activity in each subject through a signal derived exclusively from the rest of the subjects’ data”: This statement is vague enough to be interpreted in different ways (as there are multiple components in the pipeline that could be calculated over one or multiple subjects). I would suggest clarifying things with the corresponding variable names from equations (1-5).
* On line 224, the methods description switches from computation of variables related to the target region of interest (e.g. $z_{max}$), to the introduction of the specific reference image and its manipulations $z_I$. It would help to state this explicitly.
* Table 1 provides structural metrics for manipulated images together with a lower baseline computed from random images. I would suggest adding an upper baseline calculated from diffusion-based image variations without brain signal conditioning.
* In Figure 6, it could be helpful to include the reference image (e.g. in the top-left corner).
* The literature review on “optimal visual stimulus generation” misses (at least) one reference from the BrainDiffuser paper (Ozcelik et al, 2023).

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new model BrainACTIV to generate synthetic images optimizing specific brain responses from specific areas. Their model incorporates recent works on diffusion models conditioning to generate images that vary structurally with brain signal for better interpretations of brain representations and downstream analysis.

### Strengths
The paper is novel in tackling a common interpretability problems faced among works of synthesizing image based on brain signal. These synthetic images usually requires a lot of guesswork to confirm they are representing information from the brain. The IP-Adaptor is a clever way to induce structural biases into diffusional models so that the generated image vary reliably on the axis that encode the most salient representation that a brain areas cares about. This model has great potential to be extended to different brain data modality and it seems like it only requires a CLIP based brain encoding model for adaptation.

The second part of the paper analyzing mid-level features provides good examples of potentially making inferences from these synthetically generated images, further making use of the interpretability of the method.

### Weaknesses
The paper should perhaps incorporate predictive performance the Dino-ViT and fwRF encoder of these brain areas in Figure 3b so reader could have a better sense of how big of a difference the manipulation of images could make.

My main critique of this paper is perhaps more high level. Now that we have a model that could generate very clean optimized images for specific brain areas, can one use it for making new discoveries/inferences about how the brain represent visual information? Most of the work that synthesize images with brain responses has been proving that model can generate what we already know about the brain (e.g. Face image with FFA data) but rarely do they inform what we don’t already know. I am aware this paper might want to establish itself as a proof that this method could work in generating testable hypothesis but I am not yet convinced that it is substantially different than all the other papers in this subfield.

### Questions
line 323: what's the reasoning behind choosing images with responses closest to baseline activation?
Table 1: it might help to show with image what you means by "preserve structural fidelity" since "preserved" can be a relative concept. It might even be helpful to compute the numbers for images synthesized from other papers for comparison.

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
At its core, the methodology employs a CLIP-based brain encoder that transforms visual inputs into corresponding brain activation measured with fMRI, with the goal to generate interpretable images that excite or inhibit certain brain areas. The proposed method, BrainACTIVE, allows for incorporating semantic information from modulation embeddings as well as lower-level image information directly into the generation process by seeding a diffusion process specific target images. The authors validate their proposed model by training separate encoders to predict the brain's activity to the synthesized images.

### Strengths
The paper overall is very well written, clear, with a straightforward exposition and extensive experiements. Sufficient detail is provided to understand all key components of the data, model architecture, and training.

### Weaknesses
While the paper is very well presented, I'm taking issue with the related work section, the selection of experiments, and the overall novelty. Considered as a whole, the paper can be improved significantly by addressing these concerns.


**Major Concerns**
- *Related works*: there is a wealth of articles surrounding generative models for optimizing stimuli for neuronal data, especially using fMRI, see related literature (1, 2, 3, 4). ]The focus of the related work sections seems to lie on category selectivity in visual cortex and and diffusion models using CLIP. More emphasis on the wealth of related work could be placed to compare the proposed method to other approaches

- *Comparing predictive performance*: Having separate encoding models is a clear strength of this paper. However, the predictive performance of the CLIP encoder model was not convincingly shown. Showing that also the CLIP-based encoding model has a comparable performance on the held-out test set (as shown in Fig. 9) as the DINO and fwRF would provide valuable insight.  A similarly interesting analysis would be including the CLIP-encoding model into Fig. 3 to understand the change in neuronal activity from the generating model. Lastly, it is important to demonstrate how much of the variations in the images is due to the random seed in the generator models - by training different models and measuring the similarity of the optimal images, and by measuring the variability of activations in the evaluator models (DINO and fwRF)

- *Analyses of mid-level images features*: This analysis shows much promise but doesn't yield much insight. To establish the validity of these measures, it would have been important to show that the metrics are consistent for either handpicked or parametric stimuli. Similarly, these metrics could be very useful to study early- to mid-level ROIs.

- *Overall novelty*: Taken together, while this approach is promising, it doesn't deliver a new insight into either category selectivity of mid-level image feature processing of visual cortex. The analyses are mostly confirmatory with respect to the current state of understanding of visual processing. The authors could take one of the so called "no-mans-land" ROIs and discover novel categories that drive these regions, such as anterior IT.


Literature:
- (1) https://medarc-ai.github.io/mindeye

- (2) https://mind-vis.github.io

- (3)  https://arxiv.org/abs/2305.10135

- (4)  https://proceedings.neurips.cc/paper_files/paper/2023/hash/67226725b09ca9363637f63f85ed4bba-Abstract-Conference.html

### Questions
- The usefulness of the top-nouns analysis is unclear to me, and the authors don't discuss the outcome of these results in the discussion section. In general, applying BrainACTIV seems very promising when applying it to areas in the visual hierarchy that are less well understood, going beyond largely confirmatory analyses. It would be great if the authors could include further discussion points how the top-noun analysis could best be utilized in the discovery of new feature or category selectivities.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
In this paper, the authors present a method for generating maximizing or minimizing images for a brain ROI using Stable Diffusion. The method uses the NSD dataset, and builds on previous work using this dataset and diffusion models to generate activating images. Specifically, starting from a source image, the method changes this image to maximize or minimize the activation in an ROI (though the estimation of a CLIP vector characterizing the ROI from an encoding model). This makes it possible to detect the change in category representations as different regions are maximized. The method can also be used to contrast two ROIs, even ones that process the same category.

### Strengths
- The method nicely builds up on previous work (which it acknowledges well). 
- It allows a more concrete evaluation of images across ROIs due to the presence of the common starting image. 
- The formulation allows for both maximization and minimization.
- It is possible to contrast multiple ROIs along the same brain pathway. 
- The results makes sense in terms of the discovered selectivities.

### Weaknesses
1- More quantification of the observed effects is needed, or at least more details about them. 
2- More should be discussed in terms of the meaning of the maximizing/minimizing categories and the results. It is not clear what are novel  results about the brain and what is a replication.
3- The most novel results appear to be in figure 7. However these are not discussed in terms of significance or hypotheses.
4- Luo et al 2024 is used in the methods but doesn't figure in the related work section. How does the current work compare with the method used there?

### Questions
- Regarding Weakness 1: How many images are used to compute the selectivity results (e.g. figure 4), why are those chosen? What is the predicted activity associated with the maximizing and minimizing images? What are the statistics related to these activities? Could it be used to make predictions about how such images would activate or deactivate brain regions?
- Regarding Weakness 2: how do we understand the minimizing categories? Is it just a factor of having to step away from the preferred categories or is there truly a reduction in activity from baseline? Is this supported in the literature?
- Regarding Weakness 3: What hypotheses can be predicted from the results comparing two regions?

### Soundness
3

### Presentation
4

### Contribution
3

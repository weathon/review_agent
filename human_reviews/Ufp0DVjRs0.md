# Feature Accentuation: Explaining 'what' features respond to in natural images

- Decision: Reject
- Scores: 6, 8, 6, 3

## Abstract
Efforts to decode neural network vision models necessitate a comprehensive grasp of both the spatial and semantic facets governing feature responses within images. Most research has primarily centered around attribution methods, which provide explanations in the form of heatmaps, showing 'where' the model directs its attention for a given feature. However, grasping 'where' alone falls short, as numerous studies have highlighted the limitations of those methods and the necessity to understand 'what' the model has recognized at the focal point of its attention. In parallel, 'Feature visualization' offers another avenue for interpreting neural network features. This approach synthesizes an optimal image through gradient ascent, providing clearer insights into 'what' features respond to. However, feature visualizations only provide one global explanation per feature; they do not explain why features activate for particular images. In this work, we introduce a new method to the interpretability tool-kit, 'feature accentuation', which is capable of conveying both 'where' and 'what' in arbitrary input images induces a feature's response. At its core, feature accentuation is image-seeded (rather than noise-seeded) feature visualization. We find a particular combination of parameterization, augmentation, and regularization yields naturalistic visualizations that resemble the seed image and target feature simultaneously. Furthermore, we validate these accentuations are processed along a natural circuit by the model. We make our precise implementation of 'feature accentuation' available to the community as the 'Faccent' library, an extension of the popular 'Lucent' library for feature visualization.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
Previous methods focus on either where a model attends or what concept the model is looking for. This paper presents a method that can show both where and what a model is focused on -- this new method is called "feature accentuation". Feature accentuation is tested for natural-ness of images.

### Strengths
-  The paper is well-presented, along with a large number of visualizations littered both through the main text as well as the supplementary. The method is clearly explained in detail, as well as motivated explicitly each step of the way.
- The paper clearly proves what it sets out to -- namely, that the visualizations it produces highlight "where" and "what" in a natural-looking way.
- The evidence of natural-looking visualization is strong, and I'm convinced that feature accentuation can produce relatively realistic transformations, at least relative to previous deepdream-esque variants.

### Weaknesses
- My main concern is I'm not sure what the utility of feature accentuation is. A good chunk of the experiments focuses on natural-looking images, but I'm slightly less concerned about that. It's certainly a desirable property -- to look more natural and less like hallucinations. However, I'm not sure how this helps us, concretely: Can this help us improve accuracy? Diagnose mistakes? (This is suggested in several of the figures) Suggest a path for fixing the model? (Maybe by pruning "wrong" nodes?) There's a preview of this in figures 9 and 10, but the utility isn't studied in the paper. [1] for example has several approaches for defining and quantifying interpretability/utility of a method.
- Along the lines of the above, I'm not sure how to use the information these visualizations provide. For example, in figure 9, for "bow" vs. "chainsaw", I can certainly see that the "chainsaw" visualization is more chainsaw-like, but couldn't I just pull a random other word and visualize that? For example, I could feature accentuate "matchsticks" or "juggler fire torches", and I'm not sure how any of those visualizations would help me diagnose mistakes in the original model. Another way to put this would be: If I feature accentuate something clearly unrelated, eg.., "snail" for the "bow" image, it seems like this method would find *some way to insert a snail, but what does that tell me exactly, if I can insert any concept into the image? (One possibility is that natural looking insertions are "valid" explanations, and grotesque abstract art is unrelated? This would need some more fleshing out though, but that would be one way). On a side note, the main utility in figures 9 and 10 is ironically that it highlights some part of the original image, like saliency maps. (I'm not sure how important it is to be able to modify the image, per the above). A study would probably need to show that the image modification is helpful too.
- Have you tried the sanity checks in [2]? I know [2] is actually cited in your related works. Although feature accentuation is not a saliency map per se, the randomized tests in that paper should still apply. It would help (partially) my concerns above if the method does pass sanity checks -- in that it shows there *is meaning.

[1] Poursabzi-Sangdeh, et al. Manipulating and Measuring Model Interpretability. https://arxiv.org/abs/1802.07810
[2] Adebayo, et al. Sanity Checks for Saliency Maps. https://arxiv.org/abs/1810.03292

### Questions
I've left my questions above. In summary, I'm not convinced of the technique's utility, but I'm open to being convinced. The approach is certainly thoroughly explored and visually appealing. I'm just afraid that visual appeal could be a misleading objective.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new method for generating visualisations of the features leading to specific activity of neural network models, termed Feature Accentuation.
The primary novelty here is to optimise a new image such that a weighted tradeoff is generated between maximising the activation of the unit under study while staying close to the original image.
Various techniques are used (image parameterization, augmentation) to make this work.
The results of this process look generally appealing, and for certain examples quite compelling, in showing how the technique can swap class labels with subtle or not-so-subtle but still sensible image changes. 
It is also appealing that the technique can be applied without needing to use an auxiliary generative model.

### Strengths
My initial recommendation is to accept the paper, since it appears to solve (or at least, offer paths to solution) for a core problem in explainability / network visualisation.
I have some minor comments on the paper.
However, I must caveat this by saying that I have not worked in this area for several years, and so my knowledge of the literature is outdated. 
It is possible that other reviewers are aware of work that undermines the novelty or contribution of this approach.

### Weaknesses
- The paper is sloppily formatted (parentheses missing from citations, missing references, etc).
- Unpack the equation for $z^{*}$ (top of page 4) into words, since it's the key equation of the paper.
- raster plots in figure 8 are poor quality; hard to see detail.
- Figure 8 B, C: lambda should probably be 1.0 not 10.0
- Figure 10 caption should make explicit what the rows and columns are.

### Questions
- How novel is the frequency domain parameterisation? citations to other literature should be provided.
- What does it mean to achieve higher correlations in circuit similarity than natural images themselves? In a positive view, this could mean that the feature accentuation technique is settling onto good *prototypes* that provide a coherent illustration of the core concept of the label, and thereby reduces variance from natural depictions of the concept. A less positive view could be that this means the technique is finding local minima that will fail to generalize.
- Figure 8: what are the smooth curves, and how were their hyperparameters chosen? The underlying data are very noisy, so the choice of smoothing and its associated uncertainty should be reported.

### Soundness
4 excellent

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The paper introduces a model explanation technique that aims to amplify image features for a discriminative model. Specifically, the method finds visualizations that resemble an input image and a target feature, without relying on external models. The paper demonstrates the effectiveness of the framework via qualitative examples and showcases multiple applications.

### Strengths
* The approach is well-motivated and well-explained, starting from an overall objective and then the necessary additional components to address challenges encountered.
* The fact that the proposed approach does not rely on external models makes the framework a standalone explainability framework that probes the internal knowledge within one model only.

### Weaknesses
* All results in the main paper are quantitative and are shown only on a few selected examples. 
* The framework is sensitive to hyperparameters such as learning rates and regularization weights, as noted in Appendix C-D, which makes it challenging to adapt to different models.

### Questions
* Multiple qualitative examples are shown but the analysis is lacking. For example, how to interpret the results from Figure 12 in the Appendix?

### Soundness
2 fair

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
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
this paper proposes to add the class at the input of the enconder of an ecoder-decoder transformer model and claim that this leads to an interpretable classifier based on transformer architecture that is suited for interpretable fine-grained cassification.

### Strengths
The simplicity of the the method is the main strength of the paper.
It reports results in a plethora of datasets to showcase the soundness of the method.

### Weaknesses
Comparison the ResNet model is not relevant in this set up. The comparison should be with the same architecture without the class query at the decoder and other similar architectures changing the relevant parameters.
The relevance of interpretability is arguable because it is not quantified. It is also not clear how different this is with the typical existent transformer architectures. A more meaningful comparison would strengthen the paper contribution.

### Questions
See weaknesses points.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

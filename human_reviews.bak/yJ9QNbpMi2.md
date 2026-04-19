# Brain Mapping with Dense Features: Grounding Cortical Semantic Selectivity in Natural Images With Vision Transformers

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6, 6

## Abstract
We introduce BrainSAIL (Semantic Attribution and Image Localization), a method for linking neural selectivity with spatially distributed semantic visual concepts in natural scenes. BrainSAIL leverages recent advances in large-scale artificial neural networks, using them to provide insights into the functional topology of the brain. To overcome the challenge presented by the co-occurrence of multiple categories in natural images, BrainSAIL exploits semantically consistent, dense spatial features from pre-trained vision models, building upon their demonstrated ability to robustly predict neural activity. This method derives clean, spatially dense embeddings without requiring any additional training, and employs a novel denoising process that leverages the semantic consistency of images under random augmentations. By unifying the space of whole-image embeddings and dense visual features and then applying voxel-wise encoding models to these features, we enable the identification of specific subregions of each image which drive selectivity patterns in different areas of the higher visual cortex. This provides a powerful tool for dissecting the neural mechanisms that underlie semantic visual processing for natural images. We validate BrainSAIL on cortical regions with known category selectivity, demonstrating its ability to accurately localize and disentangle selectivity to diverse visual concepts. Next, we demonstrate BrainSAIL's ability to characterize high-level visual selectivity to scene properties and low-level visual features such as depth, luminance, and saturation, providing insights into the encoding of complex visual information. Finally, we use BrainSAIL to directly compare the feature selectivity of different brain encoding models across different regions of interest in visual cortex. Our innovative method paves the way for significant advances in mapping and decomposing high-level visual representations in the human brain.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces a method for mapping the brain activity obtained using fMRI to semantically meaningful features in image space using vision transformer backbones and a parameter-free distillation process for denoising.

### Strengths
- The method was explained clearly
- Literature review seems exhaustive
- Claims SOTA (although I couldn't verify this)

### Weaknesses
-Abstract strats with features of BrainSAIL (first few sentences) before getting into WHAT BrainSAIL is.

-It is not clear to me after training the backbone and distillation, how much of the output result is actually driven by brain data or completely rely on features of the image learned by the frozen foundation model.

-SOTA was claimed but I wasn't able to find any comparison to other relevant methods in the paper.

### Questions
How is it possible to verify that the method is not dominated by the foundation model representation of image rather than actually representing brain semantic representations?

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
4

### Summary
The authors proposed a method called BrainSAIL to deal with the problem of multiple categories co-existing in the natural images, combining the image embeddings and dense visual features and further identifying specific sub-regions of each image, which reflects the selectivity patterns in different areas of the higher visual cortex. BrainSAIL can realize the semantic attributes and positioning of related objects in complex images, which helps decompose the high-level visual representation of the human brain.

### Strengths
1.	BrainSAIL can obtain dense embeddings without artifacts and additional training through an efficient learning distillation module for high-throughput presentation of the visual cortex on large datasets.
2.	Compared to other studies that used simplified images, BrainSAIL can isolate specific image regions that activate different cortical areas when viewing natural scenes, providing a more complete description of how real-world visual stimuli are represented and processed.
3.	The experiments are very sufficient, especially in the appendix.
4.	This work demonstrated that an artifact-free dense feature map can be derived to explore high-throughput selectivity in the visual cortex.

### Weaknesses
1.	This method has a high model dependency (e.g., CLIP, DINO, SigLIP). While this improves efficiency, it limits flexibility to adapt feature extraction to the specific needs of neural regions, potentially constraining the granularity of semantic selectivity.
2.	The method's fMRI training uses specific datasets (e.g., NSD), which may introduce bias, limiting generalizability across different populations or visual tasks—especially when the dataset’s images or semantic information don't cover the full range of possible visual stimuli.

### Questions
DINO trained without language guidance shows high sensitivity to low-level visual features, does this hurt its performance on high-level semantic selection?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper aims to explore the semantic topography in human visual cortex using fMRI. The main goal is to disambiguate the typically confounded nature of multiple categories which is a feature/issue in naturalistic images. The trend towards naturalistic datasets in recent years necessitates more advanced methods to be able to deal with such confounds and this work takes a good step forward in the right direction. This paper then goes on to look at what regions of human visual cortex are driven by low-level image features, providing a complementary view into how the authors’ method can be used beyond studying categorical selectivity. This method provides a good solution to explore such potential confounding effects in determining what is driven by high or low level visual information.

It was a very nice paper to read and shows the authors' clear expertise and attention to detail in terms of the questions we should be asking in the community.

### Strengths
- The authors show alternatives to high-frequency artefact removal that are more computationally efficient than other solutions out there, namely by recognising the existing common method of adding register tokens in ViTs can be replaced with their proposed learning-free distillation module, which avoids the need for (computationally demanding) additional training.
- The authors show that this method is a good way to target high confounds in naturalistic images by studying things like colour confounding with food images, luminance modulations in inside/outside scenes, depth information in varying images containing varying “reach spaces” etc.
- The paper highlights potential equivalence between representations derived under varying training algorithms and questions what is being learned and the generalisation of features. I found this to be timely as recent discussions with LLMs and the utility of methods like BrainScore have been brought into question. I think we need to be asking these questions across the board and I appreciate this final section of the paper that raises this issue.
- Visualisations are very nice and clear, making some more computationally dense text a bit more clear with reference to specific examples that underlie various findings (e.g. the effects of no language supervision in DINO and the downstream affects on the derived image maps)

### Weaknesses
- PDF is nearly 50 pages long. While ICLR allows for 10 main pages and unlimited supplementary information, the density of what is provided is a bit on the extreme side. In light of this, I still found myself searching for implementational information that I felt was missing and could be added in a revised version (see *Questions*)
- I reformulated many of the comments I first wrote as weaknesses into more directed questions in the section below, hoping that these points might be more easily addressable and will hopefully make it into the camera-ready version such that readers have a better experience of the paper by benefiting from some of the points of clarity that I ran into while reading.

### Questions
- “We validate this dense feature mapping method on a large-scale fMRI dataset consisting of human participants viewing many thousands of diverse natural images that span a wide range of semantic categories and visual statistics (Allen et al.,2022).”
    - The work by Kamitani recently showed that UMAP gave something like 40 clusters of semantic diversity at a broad level, forcing us to re-evaluate our assumed understanding of just how semantically varied NSD actually is. Do you believe the findings in the "spurious reconstructions" paper have an implication on the notion of semantic diversity as described in your work?

- The introduction has repeated references to BrainSAIL’s dense embedding framework but I’m not sure why this point is continually brought up as the vision model embeddings on the natural image are dense to begin with and I can’t see that the implementation of a step to make embeddings more dense is applied anywhere, so I was a bit lost on this point. You have the adapters and a linear probe and everything always seems to be in a dense (not non-sparse) setting. Some clarification on this point would be welcome in the paper as you seem to mean something different to how I understand the same phrasing.

- Was there supposed to be a section explaining the data choices for the brain data (surface vs voxels, subject-space vs template-space) etc.? This is kind of left to be assumed from other sections of the data but I think a small section in the supplemental outlining the data usage choices more specifically would render the procedure more complete with regard to necessary experimental details. For example, the fact you’re using the masks in Jain et al. (2023) implies working in standard (normalised) template space, but I sort of expected being able to verify this explicitly via some technical description somewhere in the supplemental materials.

- You described how you selected varying functional regions from the dataset (food masks in Jain et al. (2023) and t > 2 in the Stigliani fLoc dataset) but the selected voxels being plotted, the selection of them, is not described and also a piece of missing information. Was this taken by looking at noise ceiling values calculated in NSD and if so, what value was used as the threshold?
- Lines 427-429: you identify the areas surrounding the FFA to be selective for colour, and point out this corresponds to the food area previously identified by others. Can you elaborate on the apparent contradiction between your assumption that this is actually a food area (identified earlier in the paper) or is it one driven by high colour-sensitivity and therefore might be confounded with highly colourful food images?
- Figure 7: I don’t understand how these values are calculated and the y-axis is not labelled. I wanted to ask you to be more precise on what “spatial similarity” means in terms of the method use to derive the results.
- Not quite sure the logic for why the procedure described with the learning-free distillation module results in denoised features. Why does applying a transform, running an image through an encoder, then projecting the results back again, result in denoised features? It seems to work but I just felt like the method was explained in the "how" and not "why" it works. The paper could be made stronger by adding in some intuition in the relevant section.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
Methods:
- Embeddings from deep neural networks are used to predict fMRI recorded brain-responses.
- Features are extracted from the model and input to a denoising process. The smoothed features are then input to the fMRI encoder (I think) to create spatial attribution maps.
- UMAP is applied to the encoder parameters, which uncovers previously reported category selective regions for faces, places, words, food, and bodies.
- The spatial attribution maps are correlated with pixel depth, color saturation, and color luminance to create voxel-wise correlation values for these image properties. Place areas were found to be more correlated with depth, food areas with color, and OPA with color/luminance.

### Strengths
The paper is well written and clear. The main contribution is the use of model features to create voxel and image-wise spatial contribution maps. This is quite useful as an interpretation technique. The use of pixel-wise metrics like depth, saturation, and luminance are a powerful extension of the method.

### Weaknesses
All of the investigated category-selective areas (food, places, words, faces, bodies) are previously reported. Would be interesting so see this tested on less common or more fine-grained categories.

I found the explanation of the methods is a bit confusing (see questions). Could use clearer high-level descriptions before diving into the finer details.

I think there should be a visual comparison of attribution maps for different voxel categories with the same images. I was able to compare a few in figure 3 since there are some repeated images in words/food and face/body voxels. If the method is working then the word voxels should highlight words more than voxels for other categories. I would expect to see the face voxels highlighting faces a lot more than body parts, and the opposite for body voxels. However in the picture of the baby shared between the body/face voxels I don't see much of a difference in their attribution maps.

### Questions
1. My understanding is that the voxel-wise adapters predict a scalar brain response from an M-dimensional output of the backbone model. How is the voxel-wise adapter then applied to the dense features? I am thinking the dense features have a shape (H, W, M), and the adapter transforms the last dimension to get a scalar (H, W) activation map (figure 1.b). I found this a bit unclear in the paper.

2. The learning free distillation is applying random spatial transformations to the input image, applies the inverse transformation to the features, and then averages them all together. Is my understanding correct? 

3. I do not understand what equations 3-5 are about.

4. Figure 4 states that the UMAP basis is computed image-wise. I found it unclear how this was done, and what is the significance of an image-wise UMAP?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes a method, BrainSAIL, which aims to disentangle complex images into their semantically meaningful components and locate them to brain voxels or regions. With the prevalence of pretrained models (trained on very large amounts of data), deep learning offers a promising way to explore how semantics are organized in the brain cortex. Compared to prior methods, BrainSAIL focuses on selectivity in single-object images at the broad category level, thereby enabling a richer decomposition grounded in the full semantic complexity of natural visual experiences.

### Strengths
This is clearly written paper and makes a clear contribution. The idea of isolating specific visual features to determine selectivity effects in different cortical areas is novel and interesting. The proposed method can be used to explore the selectivity of higher visual cortex with respect to localized scene structure and image properties. This work achieves promising open vocabulary CLIP-based segmentation results.

### Weaknesses
The current experimental comparisons (open-vocabulary segmentation) are limited, which restricts a comprehensive evaluation of the proposed method. The proposed approach appears to closely resemble BrainSCUBA. Certain technical details are unclear, which makes it challenging to fully understand or reproduce the method.

### Questions
Was the same adapter used in Figures 1a and 1b? Also, were the adapter parameters frozen in Figure 1b? If they were frozen, why not use clean dense features to train the adapter?

How does the quality of the dense features impact the adapter's behavior? I suggest adding an ablation study to explore this effect.

### Soundness
3

### Presentation
3

### Contribution
3

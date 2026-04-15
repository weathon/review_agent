# Brain decoding: toward real-time reconstruction of visual perception

- Decision: Accept (poster)
- Scores: 6, 8, 8, 6

## Abstract
In the past five years, the use of generative and foundational AI systems has greatly improved the decoding of brain activity. Visual perception, in particular, can now be decoded from functional Magnetic Resonance Imaging (fMRI) with remarkable fidelity. This neuroimaging technique, however, suffers from a limited temporal resolution ($\approx$0.5\,Hz) and thus fundamentally constrains its real-time usage. Here, we propose an alternative approach based on magnetoencephalography (MEG), a neuroimaging device capable of measuring brain activity with high temporal resolution ($\approx$5,000 Hz). For this, we develop an MEG decoding model trained with both contrastive and regression objectives and consisting of three modules: i) pretrained embeddings obtained from the image, ii) an MEG module trained end-to-end and iii) a pretrained image generator. Our results are threefold: Firstly, our MEG decoder shows a 7X improvement of image-retrieval over classic linear decoders. Second, late brain responses to images are best decoded with DINOv2, a recent foundational image model. Third, image retrievals and generations both suggest that high-level visual features can be decoded from MEG signals, although the same approach applied to 7T fMRI also recovers better low-level features. Overall, these results, while preliminary, provide an important step towards the decoding - in real-time - of the visual processes continuously unfolding within the human brain.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors perform MEG conditioned visual decoding.

Compared to other works that leverage fMRI, MEG is a different information source that presents unique challenges.

The authors use an align then generate strategy, where they learn a function that takes as input the MEG signal, and train it to align with a CLIP latent using a weighted sum of infoNCE and MSE loss. For image generation, they use Versatile Diffusion and regress the needed conditioning variables from MEG. 

They observe that it is possible to recover high level semantics in the reconstructed images.

### Strengths
MEG decoding of full images is an under-explored area compared to fMRI based decoding, and it is a harder task, given the low channel count relative to the tens of thousands of voxels in fMRI. To my knowledge, this is the first time that image decoding has been demonstrated using MEG.

The paper is methodologically sound, outlining different training objectives for different parts of the proposed pipeline. The paper provides systematic benchmarks, showing that their MEG decoder leads to reasonable image retrieval and image generation.

I applaud the authors for showing "representative" retrieval and best/mean/worst decoding results, which helps gauge the effectiveness of the method.

It is also interesting that they found MEG capable of recovering high level semantics. Although it is not fully clear if this is a limitation of MEG or their method (should probably discuss more).

### Weaknesses
As with other deep decoding papers, it is not super clear what the ultimate scientific insight is. This is not a criticism specific to this paper, but more generally aimed at current decoding works which leverage powerful image priors and deep non-linear decoding/embedding functions. 

In this aspect, this paper is better than most, as their Figure 3 provides some insight on the temporal dynamics of decoding. I think it would benefit the paper to add some discussion (not necessarily experiments) on extending this to EEG based decoding, or other potential practical applications or scientific insights. 

**General clarifications:**

The clarity of many of their methods could be improved. The author repeatedly references Defossez et al. in reference to their methods. But in the main text it is not super clear. Concretely I would like the authors to clarify the following:
1. What the the MEG conv "encoder" convolving over?

2. How do you combine the MEG channels?

3. What temporal aggregation layer did you use? You mention global pooling, affine, and attention. Which layer did you end up using? Because you discuss this and then never talk about which method you ended up using.

4. How do the different aggregation layers work? I ask this question in the context of Figure 3. Because you discuss using a 1500ms window, then shift to a 250ms window. Do you train a new model? Do you re-use the 1500ms model but change the aggregation? If you do train a new model, are you taking multiple 250ms windows and supervising with the same image target? For the sliding window, what is the step size?

5. For Figure 2, in the supervised models (VGG, ResNet, etc.) are you using the last layer (1000 imagenet classes layer), or the post-pooled layer. 

6. For retrieval, are you always using cosine/dot-product similarity?

**Minor format error:**
1. The authors have ICLR 2023 in the header, when it should be ICLR 2024. And they have line numbers, which do not seem to be present in the default ICLR template.

**Minor clarifications:**
1. Can you clarify if $N$ (line 75) denotes the number of images? It doesn't seem like you define $N$ prior/after using it.
2. To provide more context, can you mention in line 84 that you are using the infoNCE loss, rather than just mentioning the CLIP loss.
3. In section 2.2, can you clarify if you are normalizing $\hat{z}$ to norm = 1 for eq. 1, and not assuming a fixed norm for eq. 2? Otherwise it seems like the two losses would have trivially the same optima, but I guess you are trying to have one loss align the direction, and have a second loss align the direction + norm.

**Additional citations:**

The author discusses one approach towards decoding, but I would appreciate if the author could also discuss the brain gradient conditioned image generation work listed below, the most recent of which also leverage GANs/Diffusion models:

Inception loops discover what excites neurons most using deep predictive models (**Nature 2019**); Neural population control via deep image synthesis (**Science 2019**); Evolving Images for Visual Neurons Using a Deep Generative Network Reveals Coding Principles and Neuronal Preferences (**Cell 2019**); Computational models of category-selective brain regions enable high-throughput tests of selectivity (**Nature Communications 2021**); NeuroGen: Activation optimized image synthesis for discovery neuroscience (**Neuroimage 2022**); Brain Diffusion for Visual Exploration: Cortical Discovery using Large Scale Generative Models (**NeurIPS 2023**); Energy Guided Diffusion for Generating Neurally Exciting Images (**NeurIPS 2023**)

Overall I think the paper is sound, interesting, and provides good insight on neural decoding from an often overlooked modality.

### Questions
Please see the weakness's section for questions.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors developed a model based on contrastive and regression objectives to decode MEG, resulting in 7X improvement in image retrieval over a classic linear decoder. The promising results in image retrieval and generation are significant in that the presented approach allows the monitoring of the unfolding of visual processing in the brain based on MEG signals, which have much higher temporal resolution than fMRI. The work yields two potentially interesting observations: (1) late responses are best decoded with DINOv2, and (2) MEG signals contain high-level features, whereas 7T fMRI allows the recovery of low-level features, though it would be worthwhile to articulate or speculate what these findings mean for understanding the cascade of visual processes in the brain.

### Strengths
The work is significant in that there is no MEG decoding study that learns end-to-end to reliably generate an open set of images. Thus, it can potentially be considered a ground-breaking in this area of research, even though the techniques used are not necessarily novel from an ML perspective.

### Weaknesses
The decoding work is supposed to provide new insights to the cascade of visual processing and the unfolding of visual perception in the brain.  The authors need to articulate better what insights the current observations (mentioned in the Summary) actually provide us.

### Questions
What do the two observations tell us about the unfolding of visual perceptual processes?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
4 excellent

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper the authors propose a method to decode brain activity. The main idea is to train an MEG decoder which maps MEG signals to a feature space which is then used to reconstruct images using a pretrained image generator. 

The authors show that MEG decoder which is a DNN leads to 7 times improvement over linear decoders which is a common approach in neuroscience studies. Image generation results suggests that it is possible to reconstruct semantically accurate from MEG activity while low-level details are difficult to reconstruct.

### Strengths
1. 7x improvement in decoding accuracy over linear decoders. This is an important result which will encourage neuroscience researchers to use DNNs for decoding MEG/fMRI signals.
2. Clear presentation of methods (Figure 1, Section2).

### Weaknesses
1. The reconstruction results are not impressive. Even the best examples shown in Figure 5 often do not have the reconstructions of image of same or related category.  Therefore, the title is misleading as the main contribution of this paper in my opinion is DNN based MEG decoder and retrieval results and is not correctly reflected in the title.
2. The decoder is trained using a combination of two loss functions : MSE loss and CLIP loss (equation 3, line 91). There seems to be no ablation study investigating what is the impact of each loss function in retrieval performance. There is one figure in supplementary material Fig S2 E but I am not sure whether it indicates two terms of CLIP loss or two terms of overall loss (CLIP + MSE).
3. In Line 110 authors mention that they select  lambda by sweeping over {0.0, 0.25, 0.5, 0.75, 1.0} and pick the model whose top-5 accuracy is the highest on the large test. Is the hyperparameter search for lambda done on test data?
4. The claim in the abstract "MEG signals primarily contain high-level visual features" does not have sufficient evidence based on the reconstruction results only. It has been shown in literature (even in Things dataset paper Figure 8) that fMRI responses of early visual cortex (which can decode low-level features) are correlated with MEG responses (Cichy et al. 2014, Hebart et al. 2023) in early time windows. Therefore, a stronger evidence is required to back this claim. A possible explanation why the reconstructions can not recover low-level details might be that temporal aggregration layers leads to suppresion of low-level features which are present in a smaller time-window around 100ms. Another possible explanation is that we are predicting a high-level feature  (DINOv2/CLIP etc.) from MEG which may not need information from low-level features and thus the image generated also lack these details. 
5. The main result of the paper is 7x improvement over linear decoders. It is not clear where exactly this result is in the paper. A reader needs to compare results in supplementary and Figure 2 in the main text. Simply adding shaded bar in Figure 2 for linear decoder next to each bars can improve clarity

### Questions
Please refer to weaknesses section for points to address in rebuttal. 

Overall this paper has some new contributions but authors make some claims which do not have sufficient support in the results. Therefore, my recommendation would be to either tone down the claims or present good evidence to back them up

### Soundness
3 good

### Presentation
4 excellent

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This contribution concerns the interesting topic of decoding/retrieval and reconstructing of visual input from MEG data (THINGS-MEG data set). The approach is based on representations of images and MEG data using multiple architectures and multiple levels of generalization.

There is a rich literature on decoding and reconstructing visual and audio stimulus from brain recordings, so novelty is somewhat limited.

Decoding is evaluated as retrieval in closed and open set conditions (the latter using zero-shot setting).
Retrieval is based on linking by learning to align MEG and image representations 

The reconstruction of visual input is based on generative models, using frameworks that have been developed elsewhere (Ozcelik and Van Rullen). 

Compared to the very rich literature on methods based on MEG and other modalities, this study has an increased focus on temporal resolution of the retrieval process and furthermore, they use diffusion models for conditional generation.

### Strengths
Compared to the very rich literature on methods based on MEG and other modalities, this study has increased focus on temporal resolution of the retrieval and furthermore using sota diffusion models for conditional generation.
It is concluded that retrieval interesting peaks following image onset and image offset (the latter based on the after-image presumably). Retrieval performance is good for several image representations (VGG and DINOv2)
The generative performance is evaluated in a number of metrics, there is good consistency among the metrics.
Visually the generation makes sense.
Useful to see examples  stratified over good, bad and ugly cases.

### Weaknesses
There is a rich literature on decoding and reconstructing visual and audio stimulus from brain recordings, so novelty is somewhat limited.

Based on MEG we have high time resolution and SNR. In the temporally resolved analysis, it is interesting that VGG outperforms the more advanced representations for the direct image (after image onset) while the more complex image representations dominate retrieval based on the after-image (following image offset). We miss a discussion of this interesting finding.

The generative performance is evaluated in a number of metrics with good consistency among the metrics. Yet, we are missing uncertainty estimates to weigh the evidence in this case

Visually the generated imagery is intriguing. However, we miss a discussion of the notable lack of fine grained semantic relatedness (generation seems primarily to pick up on texture, object scale(?) and high-level semantics eg. man-made vs natural)

### Questions
Based on MEG we have high time resolution and SNR. In the temporally resolved analysis, it is interesting that VGG outperforms the more advanced for the direct image (after onset) while the more complex image representations dominate retrieval based on the after image (after image offset). Missing a discussion of this interesting finding.

The generative performance while evaluated in a number of metrics with good consistency among the metrics. Yet, we are missing uncertainty estimates to weigh the evidence in this case

Visually the generated imagery is intriguing. However, we miss a discussion of the notable lack of fine grained semantic relatedness (generation seems primarily to pick up on texture, object scale(?) and high-level semantics eg. man-made vs natural)

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

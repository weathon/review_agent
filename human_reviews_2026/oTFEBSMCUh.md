# The hallmark of colour in EEG signals

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 6, 2

## Abstract
Our perception of the world is inherently colourful, and colour has well-documented benefits for vision: it helps us recognise objects more quickly and remember them more effectively. We hypothesised that colour is not only central to perception, but also a rich and decodable source of information in electroencephalography (EEG) signals recorded non-invasively from the scalp. Previous studies have shown that colour can be decoded from neuroimaging brain signal to simple, uniformly coloured stimuli, but it remains unclear whether this extends to natural, complex images where colour is not explicitly cued.
To investigate this, we analysed the THINGS-EEG dataset, in which 64-channel EEG was recorded while participants viewed over 1,800 Our perception of the world is inherently colourful, and colour provides well-documented benefits for vision: it helps us see things quicker and remember them better. We hypothesised that colour is not only central to perception but also a rich, decodable source of information in electroencephalography (EEG) signals recorded non-invasively from the scalp. While previous work has shown that brain activity carries colour information for simple, uniform stimuli, it remains unclear whether this extends to natural, complex images with no explicit colour cueing.
To investigate this, we analysed the THINGS EEG dataset, which contains 64-channel recordings from participants viewing 1,800 distinct objects (16,740 images) presented for 100 ms each, yielding over 82,000 trials. We established a perceptual colour ground truth through a psychophysical experiment in which participants viewed each image for 100~ms and selected the perceived colours from a 13-option palette. An artificial neural network trained to predict these scene-level colour distributions directly from EEG signals showed that colour information was robustly decodable (average F-score of 0.5).
We further examined the effect of colour features on object decoding. Using a contrastive learning framework, we modelled colour–object perception with the Segment Anything Model (SAM), in which all pixels within a segment were replaced with their average colour, followed by standard feature extraction using CLIP vision encoders. We trained an EEG encoder, CUBE (ColoUr and oBjEct decoding), to align features in both object and colour spaces. Across EEG and MEG datasets in a 200-class recognition task, incorporating colour improved decoding accuracy by approximately 5%.
Together, these findings demonstrate that EEG signals recorded during natural vision carry substantial colour information that interacts with object perception. Modelling this interaction enhances the power of neural decoding.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper aligns object and color information with contrastive learning between EEG and vision modalities. Significant performance has been achieved in object and color classification tasks. It also shows that the separate color information enhanced the object decoding.

### Strengths
The work focuses on an important task, decoding color information from EEG signals. Hundreds of participants were involved to obtain a psychophysical color ground truth. It showed significant improvement compared to other methods, and the color information significantly improved the overall performance. Several image encoders were compared for demonstration.

### Weaknesses
1. There are a few comparisons with other works focusing on color decoding. More comparison would enhance the contribution.
2. It's not clear how the color information was involved in the object decoding. Is that training the contrastive learning method with additional SAM-augmented images?
3. It seems CUBE and UBP achieve largely higher results than previous methods. What is the key increase?
4. The evidence is not convincing to demonstrate that color information was extracted from EEG signals, though the classification results are good. Figure 5 shows that the color has a significant impact on the final performance at a very early stage, where we know the visual processing is not beginning then.
5. It would be beneficial to include more analysis of the psychophysicial experiments to show the consistency between human and machine choice.

### Questions
Please see the Weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper discusses a method to extract color information about stimuli from EEG recordings of subjects viewing those stimuli. It is evaluated on the Gifford dataset. In a separate data collection effort without EEG, subjects were asked to identify the colors present in the stimuli used in the Gifford dataset. Two major results are presented. In one, an EEG decoder is trained with contrastive learning to produce embeddings aligned with the output of CLIP applied to the corresponding stimuli that have been color enhanced by segmenting the stimuli and colorizing them with the average color per segment. The output of the trained EEG decoder is passed to a color projector that produces a color distribution that is trained with binary cross entropy against the color distributions produced in the second data collection effort. In the second, the standard retieival-based method is used to evaluate the EEG decoder trained as above on color enhanced images against one trained without.

### Strengths
The central claim of this paper is that it is possible decode color of stimuli from EEG recordings of subjects viewing the stimuli. This would be important if it is substantiated. But I question this in light of my comments below.

### Weaknesses
This whole effort was conducted in the retrieval-based paradigm that is common with the Gifford dataset. Classical decoding work was formulated in either a classification or framework or a detection framework. In these frameworks, test samples are treated independently. In the retrieval-based paradigm, the entire test set is considered at once. For each test EEG sample, the closest test image sample is chosen, after both are mapped to their embeddings. Accuracy is measured by the fraction of test EEG samples for which the correct test image sample is retrieved.

The retrieval-based formulation leads to much higher "accuracies", for example, about 50% on line 294 on a 200-way zero-shot task. But this is misleading. State of the art classification accuracies with the classical classification formulation, where test samples are classified independently, are about 7% for 40-clas on EEG, even when trained with 800 samples per class. This is because the problem formulation is different. In some sense, the retrieval-based formulation is akin to classifying samples with a nearest-neighbor classifier constructed with labeled test set samples.

I would be more convinced of the ability to decode color from EEG recordings if the following simple experiments were conducted and their results reported.

 1. Show subjects screens with solid red/blue/green and perform a classical 3-way classification of EEG recordings, where stimuli are presented for 500ms to 2s. I suspect that you would get about 70% on this 3-way task, where chance is 33% if counterbalanced, because that is about state of the art on 3-way EEG classification from image stimuli presented for that long.
 2. Repeat (1) with 13 colors as done in this paper. I suspect that you would get about 17% (chance is about 7%), because that is about state of the art on 13-way EEG classification.
 3. Repeat (1 and 2) with an RSVP paradigm where stimuli are presented for only 200ms. I suspect accuracy would drop precipitously.
 4. Since you report that subjects report 2 colors per stimulus on average, either do a 13^2=169 way classification on images where the left/top half is one color and the right/bottom half is a different color, or apply a detection paradigm to these stimuli, where you train 13 color detectors, apply all of them to the EEG signal, and evaluate with ROC curves or precision/recall curves. The latter can be applied to more than two colors per stimulus. Do this first with 500ms-2s per stimulus then repeat in an RSVP framework with 200ms per stimulus. I suspect that you will get chance.

If you could get above chance with statistical significance, I would be convinced that it is possible to decode color from EEG. Short of that, I suspect that two things are going on in your results:

 1. Color is confounded with object/scene characteristics, i.e. grass is green, sky is blue.
 2. The whole thing appears to work because of the retrieval-based evaluation paradigm and would fail in a classification or detection paradigm.

### Questions
None

### Soundness
2

### Presentation
4

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the decoding of color information from EEG signals recorded during natural image viewing, and whether color modeling can improve object decoding. The authors create a perceptual color dataset from the THINGS-EEG dataset. They introduce CUBE, a contrastive-learning framework that aligns EEG representations with CLIP features from both original and color-segmented images.

### Strengths
- The paper is clearly written and well-organized.
- The construction of a colour perception dataset that complements the THINGS-EEG collection. 
- It will likely be useful not only for this study but also for future research on color representation and decoding in the brain.

### Weaknesses
- While the idea of integrating color into EEG decoding seems promising, the methodological innovation of the CUBE framework itself seems incremental compared to prior EEG decoding works based on contrastive alignment. 
- Does adding color lead to consistent improvements in all feature types or only specific object categories?
- The paper only conducts object recognition task. I am curious about whether color information can help in generation tasks (e.g., reconstructing viewed images from EEG or MEG)? 
- Some similar related works are missing. For example, Neuro3D [1] decodes the color information of 3D objects from EEG and also integrates it into the object decoding. Please include the discussion.

[1] Neuro-3D: Towards 3D Visual Decoding from EEG Signals

### Questions
Please see weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper introduces CUBE, a color-aware EEG/MEG decoding framework that aligns neural activity with CLIP features from both original images and segmented, color-simplified versions. The authors use THINGS-EEG and collect new 13-color psychophysical annotations, and they report accurate and reliable color decoding from individual seconds of EEG segments. The authors also report a consistent 5% boost in 200-way object recognition when adding the color alignment.They perform a time course analysis which suggests that color becomes decodable before object identity (under 100 ms presentation).

At the core of this work is a scientific claim: EEG signals carry substantial color information during natural vision (also emphasized in the abstract). However, this is not a new idea. Numerous prior studies have already demonstrated that color can be robustly and early decoded from EEG and MEG data. As such, the main finding largely reaffirms what is well established in the literature, using a modernized CLIP-based analysis pipeline rather than uncovering new principles of neural coding or color representation.

### Strengths
1. The general question of how color information is represented and decodable from EEG and MEG is interesting and relevant to both neuroscience and machine learning.

2. The paper is well written, clearly structured, and easy to follow. The methods are straightforward, and the figures effectively communicate the main findings.

3. The reported gains in EEG/MEG decoding when including color information are consistent across participants and analyses.

4. The newly collected color annotations for the THINGS-EEG dataset are a useful addition for the community and may support future work on color representation and decoding.

### Weaknesses
1. The title claims to identify “the hallmark of color in EEG signals,” but it never becomes clear what that hallmark actually is. The paper shows that color information is decodable and improves object recognition slightly, but that’s not the same as isolating a defining neural signature or mechanism of color processing. There is no clear description of what pattern, feature, or temporal profile qualifies as this “hallmark.” As a result, the title oversells the conceptual contribution. The paper demonstrates that color is decodable (a known result), not that it has uncovered a distinctive neural hallmark of color.

2. This leads to the next imp issue which is the significance and novelty of the report. The claim that EEG signals carry substantial color information is itself not distinctive andhas been known for a very long time. In fact, most studies in neuroscience in fact use black and white images to in fact remove the contribution of color in their decoding analyses. Prior work like from Teichmann, Bartels etc. have repeatedly shown robust color decoding from EEG and MEG signals. The study essentially reaffirms these established findings in neuroscience usinga CLIP-based regression framework and does not offer any new insight into the neural mechanisms of color processing in EEG. 

3. The study shows that color helps (perhaps, see below), but it doesn’t explain why or how. There is no analysis disentangling whether the improvement comes from early visual encoding, categorical color associations, or simple low-level luminance differences. It also remains unclear what kind of color information is driving the effect. As such the reported results feels like the first step in a larger set of analyses that should have been carried out but weren’t. It shows that color improves decoding accuracy, but stops there

4. There is an additional concern that the segmentation and uniform color fills may be driving much of the reported effect. These manipulations likely introduce additional shape and part cues that CLIP is already highly sensitive to, making it unclear whether the gains truly reflect color processing in the brain or simply stronger alignment with CLIP’s object–color priors. In other words, the framework may be capitalizing on correlations between object shape and typical color rather than decoding color information per se. Without controls that separate these factors, it’s difficult to attribute the observed improvement specifically to color.

### Questions
1. The title claims to identify “the hallmark of color in EEG signals,” but what exactly is meant by this hallmark? 

2. Given that the claim that EEG signals carry substantial color information is well established, what is the distinctive significance or novelty of this study? (the difference in timings between this study and Teichman's need to be established on the same dataset)

3. (sorry many questions under this) The study shows that adding color improves decoding accuracy, but what is the underlying reason? Does the effect stem from early visual encoding, categorical color associations, or low-level luminance differences? What kind of color information actually drives the improvement?

4. Could the segmentation and uniform color fills be driving much of the reported gain by introducing additional shape or part cues that CLIP already encodes? How do the authors disentangle true color-related effects from possible object–color priors embedded in CLIP?

### Soundness
2

### Presentation
3

### Contribution
1

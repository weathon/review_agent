

{0}------------------------------------------------

# Jersey Number Recognition with Vision-Language Models and Collage-Based Aggregation

Sebastian Csizmazia

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
sebcisz@student.ubc.ca*

Harper Kerstens

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
harperk1@student.ubc.ca*

Taksh Girdhar

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
tgirdh01@student.ubc.ca*

Salma Vikha Ainindita

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
salmavkh@student.ubc.ca*

Wenqi Guo

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
wg25r@student.ubc.ca*

Ethan Methorst

*Department of Computer Science,  
Mathematics, Physics and Statistics  
University of British Columbia  
Kelowna, Canada  
emethors@student.ubc.ca*

**Abstract**—Automatic jersey number recognition in broadcast soccer footage is challenging due to occlusion, motion blur, low resolution, and partial visibility of player numbers. This work presents a tracklet-based approach that uses a vision-language model to predict jersey numbers from multiple frames of the same player rather than from a single image. The proposed pipeline combines image super-resolution, open-vocabulary number localization, and collage-based temporal aggregation to better handle noisy real-world conditions. A pre-trained vision-language model is adapted to the task using parameter-efficient fine-tuning, enabling sequence-level reasoning while maintaining computational efficiency. The paper also includes comparisons of alternative legibility classification backbones to evaluate additional design choices. Results show that LoRA fine-tuning improves accuracy from 49.7% in the zero-shot setting to 76.41%. The collage-based formulation also reduces the number of recognition calls compared with crop-based OCR, resulting in faster recognition-stage inference at the tracklet level. Together, these results suggest that vision-language models with tracklet-level aggregation are a useful approach to jersey number recognition in sports analytics.

**Index Terms**—jersey number recognition, soccer video analysis, vision-language model, tracklet-based recognition, open-vocabulary detection, fine-tuning

## I. INTRODUCTION

Jersey number recognition is a fundamental component of modern sports analytics pipelines. Reliable identification of players enables downstream tasks such as player tracking, event detection, performance analysis, and tactical modeling. Automating this process eliminates the need for manual annotation. Automation significantly improves scalability and allows real-time detection of players, enabling real-time analysis in both professional and research settings.

Despite its importance, jersey recognition remains a challenging computer vision problem. In real-world broadcast

footage, players are frequently subject to occlusions from other players, partial or complete lack of visibility of their jersey number due to camera framing, and perspective distortions caused by varying viewing angles. Additional factors such as motion blur, lighting variation, low image resolution, and jersey deformation further degrade visual quality. These challenges make it difficult to consistently extract clear and discriminative representations of jersey numbers from individual frames.

To address these challenges, this project explores a vision-language modeling approach that leverages temporal information within tracklets. Instead of relying on a single frame, multiple cropped images of a player are aggregated and provided as input to a vision-language model, which is tasked with predicting the most frequent jersey number across the sequence. This formulation allows the model to integrate information across frames, increasing robustness to noise and partial visibility.

The proposed solution builds upon a pre-trained vision-language model and applies parameter-efficient fine tuning to adapt it to the jersey recognition task. By combining visual understanding with structured prompting and sequence-level reasoning, the system aims to improve prediction accuracy while remaining computation efficient for inference. Overall, this work demonstrates how deep learning and vision-language models can be applied to a complex, real-world sports analytics problem, highlighting both their potential and their limitations in structured visual reasoning tasks.

## II. RELATED WORK

Jersey number recognition is a fundamental problem in sports video analysis because it supports downstream tasks such as player identification, tracking, analysis, and a broader

{1}------------------------------------------------

understanding of game footage [1]. However in a broadcast setting, recognition is still required because the jersey number can usually be seen only in small parts of the frame, and is damaged by motion blur, occlusion, pose variations, perspective distortion, jersey folding, and low effective resolution [1] [2]. Previous work has addressed the problem both at the image and tracklet levels, using methods ranging from specialized CNN classifiers to space-temporal and text recognition pipelines [3] [4] [5] [1].

Early image-level approaches primarily dealt with jersey number recognition as a special visual classification problem. CNN-based recognition significantly outperforms hand-crafted baselines on soccer jersey crops and helps to establish convolutional classification as an effective starting point [3]. Subsequent research improved this by incorporating more localization and supervisory techniques. The combination of holistic jersey label predictions and its components leads to better performance than the use of a single representation [5], while pose-guided R-CNN-style localization was used to better associate person regions and numerical regions in chaotic sports scenes [6]. Localization with lightweight CNN classification and synthetic pre-training was similarly used to address the scarcity and imbalance of real jersey data sets [7]. Together, these methods established the importance of three recurring concepts, good localization, resilience to low-quality crops and supervision that reflects the structure of jersey numbers, rather than treating each class independently.

Tracklet-level methods extend this line of work by exploiting temporal redundancy across frames. Since only a subset of frames in a player tracklet may clearly reveal the jersey number, several systems explicitly filter frames before recognition. Prior methods have approached tracklet recognition through legibility filtering, keyframe identification, temporal aggregation, and transformer-based sequence modeling [1]. Their 2024 framework is the baseline for our work because it combines a fine-tuned ResNet34 legibility classifier, pose-based torso localization, and a PARSeq scene text recognizer, then aggregates image-level predictions to produce tracklet-level labels. This baseline demonstrates that jersey number recognition can be treated not only as a closed-set sports classification problem but also as a structured recognition pipeline with separate stages for filtering, localization, reading, and consolidation [1].

Our method builds directly on this baseline direction, but departs from it in several important ways. First, while the baseline relies on CNN-based legibility classification, our preliminary comparison of ResNet34, EfficientNetV2, and MobileNet showed that all three models produced low average legibility scores on the selected SoccerNet tracklets, motivating a move away from standard lightweight or conventional convolutional backbones for this stage. We therefore explored more modern visual backbones, including both Swin Transformer and ConvNeXt, before adopting a transformer-based direction for legibility modeling. This choice is consistent with the broader computer vision literature. Swin Transformer was designed as a hierarchical vision transformer with shifted local windows,

giving it linear complexity with image size and making it suitable for dense visual tasks where scale variation and high-resolution structure matter [8]. ConvNeXt, in contrast, shows that a modernized pure ConvNet can recover many of the gains associated with transformer-era design by incorporating larger kernels, inverted bottlenecks, and updated training practices while retaining a convolutional structure [9]. Since jersey number visibility depends heavily on fine local structure, scale sensitivity, and robustness to degraded crops, these newer backbone families provide a more suitable alternative to a fixed CNN legibility stage than earlier architectures such as ResNet34, EfficientNetV2, or MobileNet.

Second, our pipeline puts more focus on improving crop quality before recognition. One of the biggest problems with SoccerNet is that player crops are low-resolution. Our report shows that Real-ESRGAN sharpening makes jersey numbers easier to read, both for individual crops and for the collage representation used later on. More generally, super-resolution research has shown that ESRGAN improves perceptual quality and gets sharper, more realistic textures than SRGAN by making changes to the architecture and loss levels around RRDB blocks and using adversarial training [10]. For recognizing jersey numbers, this is helpful not as an end goal, but as a way to give the downstream detector and recognizer clearer digit boundaries.

Third, we move away from using pose estimation as a stand-in for number location and toward direct number localization. The baseline and related sports systems often use pose information to figure out where the torso is and then send a crop to the recognizer [6] [1]. Instead, our pipeline uses an open-vocabulary detector to crop the number more closely and sees failed detection as a sign that the text is hard to read. The broader open-set detection literature backs this change. For example, Grounding DINO shows that language-guided detection can find any text-specified objects by using stronger cross-modal fusion [11]. Our implementation uses OWLv2 instead of Grounding DINO, but both are part of the same larger trend of replacing fixed-category detection with flexible text-guided localization. In our case, this makes the detector line up better with what needs to be seen, i.e. the number itself, not the player’s pose.

Finally, our work also departs from prior scene-text-based jersey pipelines at the recognition stage. Jersey number recognition has been framed as a constrained form of scene text recognition, and PARSeq has been shown to integrate successfully into a sports pipeline [1]. However, our current system moves from frame-wise OCR toward a tracklet-level collage representation paired with a Qwen-family vision-language model. This better matches the formulation introduced in our paper, where multiple cropped images from the same player are aggregated and passed to a vision-language model for sequence-level prediction. Qwen3-VL is relevant here because it is designed for multimodal reasoning across images and video, includes OCR-oriented training data, and extends contextual processing to longer multimodal inputs [12]. This makes a vision-language model a plausible alternative to a

{2}------------------------------------------------

rigid two-digit OCR stage, especially when ambiguity must be resolved across multiple noisy crops rather than a single clean image.

## III. METHODOLOGY

### A. Legibility Model Comparisons

1) *CNN-Based Baselines:* As part of our evaluation of the baseline pipeline’s legibility classification stage, we compared three backbone architectures: ResNet34, EfficientNetV2, and MobileNet. Each model was evaluated using the pretrained legibility classifier weights on the first five tracklets of the test set, computing the average legibility confidence score and the number of frames classified as legible (above the 0.5 threshold) across all frames.

**ResNet34** is the architecture used in the original baseline pipeline, specifically fine-tuned for SoccerNet jersey legibility by the original authors [1]. As shown in Figure 1, ResNet34 produced average legibility scores ranging from 0.387 to 0.483 on the five tracklets, with all scores falling below the 0.5 threshold. It classified the fewest frames as legible per tracklet (28–166 frames), suggesting a stricter classification boundary that may reduce false positives at the cost of missing some legible frames.

![Figure 1: ResNet34 legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.](af191c691aa1282801a350dbb14c7925_img.jpg)

| Tracklet | Average Legibility Score | Legible Frames | Illegible Frames |
|-|-|-|-|
| T1 | 0.387 | 57 | 468 |
| T2 | 0.420 | 138 | 344 |
| T3 | 0.483 | 166 | 491 |
| T4 | 0.412 | 28 | 389 |
| T5 | 0.412 | 33 | 414 |

Figure 1: ResNet34 legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.

Fig. 1. ResNet34 legibility classification results for the first five test tracklets. Top: average legibility score per tracklet. Bottom: legible vs. illegible frame counts.

**EfficientNetV2** achieved per-tracklet scores between 0.452 and 0.496, and the narrowest range of the three models and the closest to the 0.5 threshold. As shown in Figure 2, it classified substantially more frames as legible than ResNet34 (66–261 frames per tracklet), indicating a more allowing classification boundary.

![Figure 2: EfficientNetV2 legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.](e3921a931e5c1e184cf30effc70ded74_img.jpg)

| Tracklet | Average Legibility Score | Legible Frames | Illegible Frames |
|-|-|-|-|
| T1 | 0.452 | 177 | 349 |
| T2 | 0.472 | 76 | 244 |
| T3 | 0.496 | 261 | 448 |
| T4 | 0.452 | 46 | 249 |
| T5 | 0.472 | 148 | 407 |

Figure 2: EfficientNetV2 legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.

Fig. 2. EfficientNetV2 legibility classification results for the first five test tracklets. Top: average legibility score per tracklet. Bottom: legible vs. illegible frame counts.

**MobileNet** produced scores between 0.409 and 0.498, showing the highest variance between tracklets. As shown in Figure 3, the number of legible frames per tracklet varied dramatically (2–262), with tracklet T4 having only 2 legible frames compared to 262 for T5. this inconsistency suggests that MobileNet is less stable for this task.

![Figure 3: MobileNet legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.](2b37b4c11c25b2f26358c3d686ed3441_img.jpg)

| Tracklet | Average Legibility Score | Legible Frames | Illegible Frames |
|-|-|-|-|
| T1 | 0.432 | 146 | 378 |
| T2 | 0.428 | 72 | 291 |
| T3 | 0.434 | 46 | 405 |
| T4 | 0.409 | 2 | 287 |
| T5 | 0.498 | 262 | 348 |

Figure 3: MobileNet legibility classification results for the first five test tracklets. The top chart shows average legibility scores per tracklet (T1-T5) with a 0.5 threshold line. The bottom chart shows the number of legible (green) and illegible (red) frames per tracklet.

Fig. 3. MobileNet legibility classification results for the first five test tracklets. Top: average legibility score per tracklet. Bottom: legible vs. illegible frame counts.

The general comparison between all three models is shown in Figure 4. EfficientNetV2 achieved the highest overall average legibility at 0.478, followed by MobileNet at 0.458 and ResNet34 at 0.434. However, all three models scored below the 0.5 decision threshold on average, indicating that the majority frames in these tracklets are classified as illegible regardless of architecture. Despite having the lowest average score, ResNet34 remains the appropriate choice for the baseline pipeline due to its domain-specific fine-tuning and more conservative classification behaviour.

These consistently low scores across all three CNN-based models motivated us to look beyond conventional convolutional backbones. We therefore turned to more modern

{3}------------------------------------------------

architectures, first evaluating Swin Transformer (Swin-T) and then ConvNeXt, as discussed in the following sections.

![Figure 4: Overall legibility model comparison. Top: per-tracklet average scores for ResNet34 vs EfficientNetV2 vs MobileNet. Middle: legible frame counts per tracklet. Bottom: overall average legibility score across all tracklets.](e94f3bbb6f7501b9a1344dd0210e5dd8_img.jpg)

Figure 4 consists of three bar charts. The top chart, titled 'Model Comparison: ResNet34 vs EfficientNetV2 vs MobileNet', shows 'Average Legibility Score per Tracklet' for five tracklets (T1-T5). The middle chart, titled 'Number of Legible Frames per Tracklet', shows the count of legible frames for the same tracklets. The bottom chart, titled 'Overall Model Comparison (All Tracklets)', shows the overall average legibility score for ResNet34 (0.4538), EfficientNetV2 (0.4776), and MobileNet (0.4575). A dashed red line at 0.5 indicates the threshold.

Figure 4: Overall legibility model comparison. Top: per-tracklet average scores for ResNet34 vs EfficientNetV2 vs MobileNet. Middle: legible frame counts per tracklet. Bottom: overall average legibility score across all tracklets.

Fig. 4. Overall legibility model comparison. Top: per-tracklet average scores. Middle: legible frame counts per tracklet. Bottom: overall average legibility score across all tracklets.

**2) Transformer and Modern Architecture Alternatives:** We fine-tuned a ConvNeXt-tiny on SoccerNet dataset using weak supervision derived from tracklet annotations. In this setup, any tracklet with a valid jersey-number annotation was treated as legible, while tracklets labeled  $-1$  were treated as illegible. Because SoccerNet provides labels at the tracklet level rather than the frame level, each image in a tracklet was assigned the same binary legibility label. This enabled image-level training, but also introduced label noise, since some frames within a legible tracklet may still be unreadable. We therefore used this setup to train an image-level legibility classifier while evaluating its behavior at both the image and tracklet levels.

To reduce noise before training, we incorporated the same preprocessing artifacts used in the broader pipeline. We applied main-subject filtering using the Gaussian-filter JSON files so that only the most relevant images within each tracklet were used, and we excluded soccer-ball tracklets using the precomputed ball-track JSON files. The model was then fine-tuned on the filtered train split and evaluated on the filtered test split. During evaluation, we measured both per-image metrics and per-tracklet metrics, where a tracklet was considered legible if at least one image in that tracklet was predicted as legible.

The ConvNeXt model achieved strong image-level performance, reaching 80.9% accuracy, 82.9% precision, 94.3% recall, and 88.2% F1. At the tracklet level, it achieved 76.6% accuracy, 74.1% precision, 100.0% recall, and 85.1% F1. These results suggest that the classifier was effective at identifying legible examples. However, it produced more false-positive predictions at the tracklet level, which raised recall while lowering precision.

#### Swin-Tiny legibility module in SoccerNet jersey pipeline

![Figure 5: Overview of the Swin-Tiny legibility module in SoccerNet jersey pipeline. The diagram shows the flow from Input (Tracklet folders) through Re-ID & filtering, Swin-T legibility module, Pose estimation, Crop extraction, STR (separate stack), and Output (Per-tracklet jersey number JSON).](0c9723d1620cf51bc2b7a380ce7e23c0_img.jpg)

The diagram illustrates the SoccerNet jersey recognition pipeline. It starts with 'Input: Tracklet folders → per-frame jersey crops (RGB images)'. This is followed by 'Re-ID & filtering' which includes 'Re-ID → feature → outlier → main-subject filtering'. The 'Swin-T legibility' module is detailed in a box: 'Backbone: Swin-Tiny (patch4, window7, 224x224)', 'HF: SigmoidImageClassification 1 logit → sigmoid', 'Task: binary legibility (readable or not)', 'Preprocess: resize 224x224, imagenet normalize (+ train augmentations)', and 'Decision: score > 0.5 → legible'. Below this is a small diagram of the Swin-T architecture. The pipeline continues with 'Pose estimation: keypoints / torso → legible frames only', 'Crop extraction: jersey number ROIs', 'STR (separate stack): scene text recognition (e.g. PARSeq) → digit recognition, not Swin', and finally 'Output: Per-tracklet jersey number JSON'. A note indicates: 'Swin is used only for legibility gating. Jersey digits are read by STR, not by Swin.'

Figure 5: Overview of the Swin-Tiny legibility module in SoccerNet jersey pipeline. The diagram shows the flow from Input (Tracklet folders) through Re-ID & filtering, Swin-T legibility module, Pose estimation, Crop extraction, STR (separate stack), and Output (Per-tracklet jersey number JSON).

Fig. 5. Overview of the Swin-Tiny legibility module integrated into the SoccerNet jersey recognition pipeline. Swin-T is used only as a binary legibility filter after Re-ID and filtering, while jersey digit recognition is handled by the separate scene text recognition stack, producing the final per-tracklet jersey number output.

In order to further improve the pipeline’s legibility stage, we developed a binary classifier based on Swin-Tiny that determines if a frame is legible or illegible prior to any downstream localization or recognition stages [8]. A single-output sigmoid layer for binary prediction replaces the role of the initial classification head in the model, which is constructed from the Hugging Face *microsoft/swin-tiny-patch4-window7-224* checkpoint. As Figure 5 shows, in our configuration Swin-T is only employed for legibility filtering and the later recognition stages continue to handle jersey number recognition. Every input crop is normalized using the ImageNet mean and standard deviation and resized to  $224 \times 224$ . To increase robustness to appearance changes in broadcast footage, we used light augmentation during training, including random grayscale and color jitter. The training labels follow the SoccerNet legibility rule, which states that frames with a jersey label greater than zero are considered legible, while all other frames are considered illegible. The training split was balanced by undersampling the dominant class in order to minimize the impact of the class imbalance. The Swin-T legibility model was trained for 5 epochs using binary cross-entropy loss, SGD with learning rate 0.001 and momentum 0.9, and a StepLR scheduler with step size 7 and decay factor 0.1. During inference, the model outputs a legibility probability, and a threshold of 0.5 is used to decide whether a frame is passed to the later stages of the pipeline. This allowed Swin-T to act as a drop-in replacement for the earlier CNN-based legibility module while keeping the rest of the SoccerNet pipeline unchanged.

To compare the baseline ResNet34 legibility classifier with Swin-T, we designed a controlled image-level evaluation in which both models were tested on the same legibility task, using the same subset of SoccerNet test tracklets and the same decision threshold, so that any performance difference would come from the backbone rather than the data slice. The comparison used 20 valid tracklets from the SoccerNet test split, keeping the subset fixed and reproducible by sorting tracklet IDs when shuffling was disabled. For each selected tracklet, all image files in its folder were evaluated, giving a

{4}------------------------------------------------

total of 6,827 images in the run, with labels inherited from the tracklet annotation (label  $> 0$  = legible, label  $\leq 0$  = illegible). Both models were run on the same image paths with raw output scores, and the same 0.5 threshold was used to turn scores into binary predictions. To better understand the trade-off between conservative and aggressive legibility filtering, performance was tested using accuracy, precision, recall, and F1, as well as false-positive and false-negative rates. As presented in Figure 6, Swin-T improved accuracy, recall, and F1 over the ResNet34 baseline, whereas ResNet34 demonstrated perfect precision on this subset by producing no false positives. However, this came at the cost of extremely low recall. This trade-off is shown more clearly in Figure 7, where ResNet34 has a very high false-negative rate on legible crops, whereas Swin-T substantially reduces false negatives but accepts more false positives on illegible crops.

![Bar chart titled 'Legibility classification (image level)' comparing ResNet34 (baseline) and Swin-T (5 epochs) on the 20-SoccerNet test subset. The chart shows Accuracy, Precision, Recall, and F1 scores. ResNet34 has high precision but low recall, while Swin-T has improved recall and F1 but lower precision.](e0d425c8e4eef259e4c52d81426d93fa_img.jpg)

| Metric | ResNet34 (baseline) | Swin-T (5 epochs) |
|-|-|-|
| Accuracy | ~0.6 | ~0.85 |
| Precision | 1.0 | ~0.85 |
| Recall | ~0.4 | ~0.95 |
| F1 | ~0.55 | ~0.9 |

Bar chart titled 'Legibility classification (image level)' comparing ResNet34 (baseline) and Swin-T (5 epochs) on the 20-SoccerNet test subset. The chart shows Accuracy, Precision, Recall, and F1 scores. ResNet34 has high precision but low recall, while Swin-T has improved recall and F1 but lower precision.

Fig. 6. Image-level legibility classification results on the 20-tracklet SoccerNet test subset (6,827 crops) comparing the baseline ResNet34 model and the Swin-T model after 5 epochs of training. The bar chart reports accuracy, precision, recall, and F1 at a threshold of 0.5, where crops are treated as legible if the ground-truth jersey label is greater than 0.

![Bar chart titled 'Error rates on the same 6,827-image subset' comparing ResNet34 (baseline) and Swin-T (5 epochs) on false-positive (FP) and false-negative (FN) rates. ResNet34 has a high FN rate on legible crops, while Swin-T has a high FP rate on illegible crops.](6de7dcb072cef2388026fb0f504084b2_img.jpg)

| Metric | ResNet34 (baseline) | Swin-T (5 epochs) |
|-|-|-|
| FP rate (illegible crops) | ~0.0 | ~0.28 |
| FN rate (legible crops) | ~0.6 | ~0.02 |

Bar chart titled 'Error rates on the same 6,827-image subset' comparing ResNet34 (baseline) and Swin-T (5 epochs) on false-positive (FP) and false-negative (FN) rates. ResNet34 has a high FN rate on legible crops, while Swin-T has a high FP rate on illegible crops.

Fig. 7. Error-rate comparison for the baseline ResNet34 model and the Swin-T model on the same 6,827-image SoccerNet test subset. The chart shows the false-positive rate on illegible crops and the false-negative rate on legible crops, highlighting the trade-off between conservative filtering and improved recall.

Although Swin-T demonstrated better crop-level legibility performance, we did not use it in the final system because its

benefit did not extend to the entire jersey recognition pipeline. Due to computational constraints, the full pipeline comparison was executed on only two tracklets, which represented 24.28 % of all ground-truth tracklets assessed in that evaluation slice. In this controlled test, where all downstream stages were left unchanged and only the legibility module was changed, the Swin-T version performed 0.34% worse than the baseline ResNet34. Given this insignificant change, it did not warrant replacing the baseline with a more complex module, so we decided not to use Swin-T in the final pipeline.

## IV. SYSTEM DESIGN

Our pipeline replaces the multi-stage baseline, which relies on Re-ID filtering, legibility classification, pose estimation, torso cropping, and PARSeq scene text recognition with a more streamlined five-component architecture. As shown in Figure 8, the system chains image super-resolution, open-vocabulary object detection, collage construction, and vision-language model inference to a single end-to-end flow. Each component was selected to address specific limitations identified in the baseline: Real-ESRGAN enhances low-resolution inputs, OWLv2 consolidates detection and legibility filtering into one step, the collage replaces per-image confidence aggregation with tracklet-level visual context, and Qwen3-VL removes the character set constraints imposed by traditional scene text recognition.

![Flowchart of the Jersey Number Pipeline showing five steps: 1. Tracklet Images (Low-resolution inputs from Ground Truth), 2. Real-ESRGAN (Upscale images to enhance digit clarity), 3. OWLv2 (Open-vocab detection to identify jersey numbers), 4. Collage Builder (Combine crops into single tracklet image), 5. Qwen3-VL + LoRA (Predicted JN + LoRA to generate final output).](b4a7906eddfd40aaa750e19e56c94a8b_img.jpg)

Flowchart of the Jersey Number Pipeline showing five steps: 1. Tracklet Images (Low-resolution inputs from Ground Truth), 2. Real-ESRGAN (Upscale images to enhance digit clarity), 3. OWLv2 (Open-vocab detection to identify jersey numbers), 4. Collage Builder (Combine crops into single tracklet image), 5. Qwen3-VL + LoRA (Predicted JN + LoRA to generate final output).

Fig. 8. Overview of our Jersey Number Pipeline. Raw tracklet images are first upscaled using Real-ESRGAN ( $4\times$ ), then passed into OWLv2 for open-vocabulary jersey number detection and cropping. The cropped regions are assembled into a collage of up to 25 images per tracklet, which is fed into a fine-tuned Qwen-VL model with LoRA to predict the jersey number at tracklet level.

### A. System architecture

1) *Image Super-Resolution Pre-Processing:* Prior to number detection, we apply Real-ESRGAN as a pre-processing step to upscale low-resolution tracklet images. SoccerNet tracklet crops are often small, with jersey numbers occupying only a handful of pixels, making digit boundaries difficult to resolve. Real-ESRGAN performs  $4\times$  blind super-resolution using a Residual-in-Residual Dense Block (RRDB) network trained on synthetic degradation data [10], enhancing fine details such as digit edges without requiring paired training data or domain-specific fine-tuning.

{5}------------------------------------------------

![Figure 9: Effect of Real-ESRGAN 4x upscaling on a single tracklet image. (a) Original: A blurry image of a soccer player with the number 10 on their jersey. (b) Upscaled (4x): The same player, but the image is much sharper, and the number 10 is clearly legible.](440e59dae4772c0152116a3abd34331a_img.jpg)

Figure 9: Effect of Real-ESRGAN 4x upscaling on a single tracklet image. (a) Original: A blurry image of a soccer player with the number 10 on their jersey. (b) Upscaled (4x): The same player, but the image is much sharper, and the number 10 is clearly legible.

(a) Original (b) Upscaled (4 $\times$ )

Fig. 9. Effect of Real-ESRGAN 4 $\times$  upscaling on a single tracklet image. The jersey number 10 becomes substantially more legible after super-resolution.

Figure 9 illustrates the effect: the original tracklet image (left) contains a blurry number 10, while the upscaled output (right) shows substantially sharper digit boundaries.

![Figure 10: Collage comparison for the same tracklet. Left: collage built from original-resolution images, showing a 4x4 grid of blurry player images. Right: collage built from Real-ESRGAN upscaled images, showing a 4x4 grid of sharper player images.](bdc6095967437c168a3f2a4ff8ca38bd_img.jpg)

Figure 10: Collage comparison for the same tracklet. Left: collage built from original-resolution images, showing a 4x4 grid of blurry player images. Right: collage built from Real-ESRGAN upscaled images, showing a 4x4 grid of sharper player images.

(a) Original

(b) Upscaled (4 $\times$ )

Fig. 10. Collage comparison for the same tracklet. Left: collage built from original-resolution images. Right: collage built from Real-ESRGAN upscaled images. Digit edges and jersey text are visibly sharper in the upscaled version.

Figure 10 shows the effect at the collage level, comparing a collage built from original images (left) against one built from upscaled images (right). By upscaling before OWLv2 detection, the detector receives inputs with clearer digit structure, improving its ability to localize jersey number regions. The upscaled images are cached to disk after the first run, so the cost is paid only once per dataset split.

2) *Number Cropping and Filtering:* To crop the number out from the jersey, we used an open vocab object detector OWLv2 [13] instead of the pose detection used in our baseline. The reasons are (1) this would lead to a built-in legibility classification: if the detection model cannot detect the number, it means the number is likely not clearly visible. (2) this leads to a tighter crop and is more stable than pose detection, as key points detected by pose detection are only a proxy for the number’s location, while if we detect the numbers directly, we get only the number in the crop.

3) *Collage Input:* To account for single-digit/double-digit confusion and obstruction from other players, we built a collage as input for the OCR model. This aims to replace the image-level classification with tracklet level classification

in the original pipeline. The model directly gets an input of  $< 25$  crops in a single input. Since our OCR model (Qwen-3.5) supports arbitrary input resolution, we do not need to resize the image and thus avoid strengthening the numbers. An example of our collage is shown in Figure 11.

![Figure 11: An example of the collage with number 10. It shows a 5x5 grid of 25 small, blurry images of a soccer player, with thick blue lines separating the individual crops.](aa541b61e0c277c9c5b40e0936168cec_img.jpg)

Figure 11: An example of the collage with number 10. It shows a 5x5 grid of 25 small, blurry images of a soccer player, with thick blue lines separating the individual crops.

Fig. 11. An example of the collage with number 10.

4) *Qwen OCR model:* The main component of our pipeline is the Qwen OCR model. Being trained on Internet data with billion-level parameter size, Qwen has a large amount of world knowledge. It was also likely trained on OCR-like data to enhance its OCR ability as a general purpose model. However, training can enhance its task-specific ability and execution following for our task such that it will only output the number we need. Given the large size of the model and the size of the data set, we used LoRA [14] to avoid overfitting. The model is given a prompt asking for the most frequent number in the collage, and the output ground truth is the number itself.

### B. Algorithms used in the system

The two main components in the systems are both transformer based.

Training is performed in the classic Adam [15] optimizer.

Real-ESRGAN extends the ESRGAN architecture with a high-order degradation model that applies classical and neutral

{6}------------------------------------------------

degradations sequentially, enabling it to handle the complex, unknown degradations found in the real-world broadcast footage. We use the pretrained RealESRGAN\_x4plus model (23 RRDB blocks, 64 base features) without fine-tuning, as the synthetic degradation training generalizes well to the broadcast-quality SoccerNet images.

### C. Implementation Details and Benchmark Environment

For the number cropping and filtering component, the module is run on a server with 3 A6000 GPU (1 used), 2 socket CPU (Intel(R) Xeon(R) Gold 6258R CPU @ 2.70GHz) with 112 cores, and 754GB of System RAM. The pipeline is run without batching, and thus the processing time is long (20h for the whole dataset). The Qwen training is done on Google Colab with a G4 virtual machine with a Blackwell Pro 6000 Server Edition GPU. The final testing is also performed in the Google Colab environment.

### D. Error Analysis

1) *Single-Digit vs. Double-Digit Confusion:* The most common failure in our pipeline is the misclassification of double-digit jersey numbers as single digits, or vice-versa. When OWLv2 produces a tight crop that captures only one of a two-digit number due to occlusion or player movement, Qwen3-VL receives incomplete information and predicts a single digit. For example, jersey number 28 may be predicted as 8 if only the rightmost digit is visible in the majority of crops within the collage. This error is compounded by the collage approach. If most of the crops in the grid show a partial number, the model is biased towards the partial reading.

2) *OWLv2 Detection Failures:* OWLv2 occasionally fails to detect a jersey number region entirely, mainly in frames where the player is facing away from the camera, the jersey is heavily occluded by another player, or motion blur renders the number unreadable from the background. Since our pipeline uses OWLv2 as an implicit legibility classifier, no detection means that the frame is classified as illegible. These missed detections reduce the number of usable crops per tracklet. For tracklets where OWLv2 detects very few frames, the resulting collage does not contain enough information for reliable recognition.

3) *Real-ESRGAN Artifacts:* While upscaling generally improves digit clarity, Real-ESRGAN can introduce hallucinated textures on heavily degraded images or motion-blurred images. In cases where the original image contains a barely legible number, the upscaler may sharpen noise or jersey folds into digit-like patterns, potentially misleading OWLv2 into producing false detections or Qwen3-VL into misreading the number.

4) *Ground Truth Label Errors:* We identified multiple instances of incorrect ground truth labels within the SoccerNet dataset. These mislabels directly affect our reported accuracy, making it appear lower than the model’s true performance. For example, a tracklet labeled jersey number 6 actually shows jersey number 54, in which our model’s prediction can be arguably more reasonable than the ground truth.

5) *Aggressive Legibility Filtering:* Swin-T and ConvNeXt were less conservative than ResNet34 in their legibility decisions, allowing more frames to pass to later stages of the pipeline. This improved recall by reducing false negatives, but also introduced more false positives on noisy or borderline inputs, which could negatively affect downstream recognition performance.

## V. RESULTS & DISCUSSION

### A. Strengths of Our Approach

1) *Real-ESRGAN Pre-Processing:* The addition of Real-ESRGAN addresses a fundamental limitation of the input data: low image resolution. By upscaling tracklet images before detection, OWLv2 receives sharp inputs with a more discernible digit structure. As shown in Figure 9, digits that are nearly indistinguishable at original resolutions become clearly readable after  $4\times$  upscaling. Because the upscaled images are cached to disk, this step does not add overhead to repeated experimentation with downstream components.

2) *OWLv2:* OWLv2 aids our pipeline by collapsing multiple stages into one more cohesive step. The original pipeline utilized two separate models to achieve the effect of an image crop on the player number, the two stages being a legibility classifier and a pose detector. OWLv2 works to essentially combine these steps to achieve the same result: a concise cropped image of the player’s number.

By consolidating these detection features, OWLv2 has the added benefit of reducing communication between models in the pipeline. During our initial work setting up the pipeline, the majority of errors occurred during the communication process between models. Reducing this communication aided our development by cutting down time spent discovering and fixing these errors. These issues occurred largely due to incorrect dependency versions between the different kernels.

3) *Collage Input:* The original pipeline utilizes confidence-based metrics to help the model determine the final output of a given tracklet. This metric, as is common in deep learning systems, is initially set by the system designer and then adjusted by the model as it evaluates the images within a tracklet, adding a layer of fine-tuning to achieve more accurate results.

Our collage input approach works to eliminate the need for fine-tuning these hyperparameters. By consolidating the images in a tracklet into multiple grid-based compilations, we can offload this task to Qwen, allowing it to operate on a larger set of data when predicting an output. This removes the need for a confidence score for each individual image in a tracklet, simultaneously reducing computational overhead and the need for hyperparameter tuning.

4) *Tighter Crops:* One issue we immediately noticed when recreating the original pipeline was the pose detection model’s approach to cropping images. This model works to generate a cropped image that accounts for the player’s position and pose. However, the flaw we identified was that cropping based on the player’s pose often left a large amount of irrelevant area surrounding the number in the final image.

{7}------------------------------------------------

OWLv2 helps reduce this issue by producing a tighter crop around the player’s number, allowing us to feed more focused and relevant input into the deeper layers of the model.

5) *No Charset Limit:* The original pipeline utilizes PARSeq for its final step of text recognition. This approach constrains the output to two digits between 0 and 9. While this limitation makes sense in the context of jersey number recognition, issues arise when identifying double-digit jerseys. The model relies on a hyperparameter—similar to a confidence score—to determine whether more than one digit is present on a player’s jersey.

The benefit of utilizing Qwen for text recognition is the removal of this constraint. Since Qwen is trained on a broader dataset that encompasses a wider range of numerical patterns beyond those limited by PARSeq, it is better equipped to recognize multi-digit jersey numbers. An additional advantage of this implementation is that it expands the system’s potential use cases, removing the restriction of only recognizing up to double-digit numbers.

The Real-ESRGAN upscaling step uses the REALESRGAN\_x4plus model with fp16 precision on the A6000 GPU. Processing runs at approximately 0.1-0.5 seconds per image without tiling. For local development on Apple Silicon, fp32 precision with the MPS backend is used alongside a reduced OWLv2 batch size of 2 to accommodate memory constraints. The upscaled images are stored in an *images\_upscaled* directory alongside the originals, enabling the detection step to run independently without repeated upscaling.

### B. Limitations of Our Approach

1) *Processing Time:* A major limitation of our system is the computational power required to run it efficiently. Although we were able to train and execute the models on a server with strong hardware, the time required to run the system on a typical desktop computer would be impractical.

Additionally, the base pipeline analyzes each image individually, which allows for greater parallelization compared to our implementation. While we streamlined and condensed the pipeline, this introduced larger tasks for certain models that must now be handled by a single worker.

If we were to continue developing this pipeline, improving its parallelization would be a key area of focus, as it would significantly reduce overall processing time.

2) *Upscaling Overhead and Artifacts:* The initial Real-ESRGAN upscaling step adds significant processing time and increases disk usage substantially due to  $4\times$  resolution increase. Additionally, the model may introduce subtle hallucinated textures on heavily degraded or motion-blurred frames, potentially confusing downstream digit recognition. For tracklets where the original resolution is already fine, the upscaling provides diminishing returns. This is only for the first run.

3) *OWLv2 Introducing New Opportunity For Failure:* Although our implementation of OWLv2 effectively consolidates the pose detector and legibility classifier, the original pipeline was designed to address these tasks individually, with each

model fine-tuned for its specific purpose. OWLv2, however, was not developed solely to consolidate these tasks; rather, this capability is a byproduct of its broader functionality.

The primary issue lies in its handling of legibility classification. Since the model is not specifically trained for this task, there is a higher likelihood that it may incorrectly—and sometimes silently—classify certain tracklets as illegible that would have previously been considered legible by the original model.

4) *25-Crop Cap:* As mentioned, our final pipeline operates on a collage of images drawn from a tracklet. While this approach offers several benefits, a key drawback is the potential loss of information during analysis. Since our system constructs a grid of 25 images, longer tracklets may not fully benefit from having each individual frame processed.

If the clearest or most informative frame falls outside of the selected 25 images, the system loses a potential advantage in accurately determining the final output.

5) *OWLv2 Not Trained For This Data:* OWLv2, being a pretrained model, was trained on a diverse dataset. While this can offer advantages in the context of our system, it can also introduce certain shortcomings. In situations where a player is moving and the jersey appears misaligned or folded, the model may struggle to accurately recognize the number.

To address this limitation, the model would need to be retrained or fine-tuned on a more specialized dataset that captures these complex and irregular jersey positions.

### C. Performance & Abnormal Results

1) *Performance and Tuning:* Fine-tuning the model significantly improved our results. Before tuning our zero-shot Qwen model achieved 49.7% on a 1,840 sample set. Following two epochs of LoRA fine tuning the model achieved 76.41% accuracy, a significant improvement.

Figure 12 shows the training loss and validation over 264 steps. The graph indicates that the model converged around 100 steps. In the first 100 steps, the model’s loss dropped from 1.447 to 0.327, a 77% reduction.

Figure 13 indicates that our model is performing in line with what is expected for its mistakes. Looking at the models prediction of 4, the most common incorrect result is mistaking it for 44. The most common error pattern across the model being mistaking single- and double digit jersey numbers. As the players are moving during the images in a tracklet this contributes to the models confusion in these instances.

2) *Abnormal Results:* One initial unexpected result was the zero-shot accuracy. Although expectations were low, a result of 49% is notable poor. Upon further analysis of our pipeline and implementation, we attributed this outcome to the unique format in which the grid presents the data.

Although the grid displays the number in a concise manner, there is still a significant amount of noise and distortion within the tracklet images, which can make recognition more difficult for the model. Additionally, the model is not trained to expect this grid-based input format, which further contributes to the initial poor performance.

{8}------------------------------------------------

![Figure 12: Loss Convergence plot showing Training Loss (blue line with circles) and Validation Loss (orange line with circles) over 250 training steps. The Training Loss starts at approximately 0.325 and decreases to about 0.285. The Validation Loss starts at approximately 0.22 and decreases slightly to about 0.215.](352c5fab6f936356e9570761a02ab71e_img.jpg)

| Training Steps | Training Loss | Validation Loss |
|-|-|-|
| 100 | 0.325 | 0.220 |
| 150 | 0.295 | 0.218 |
| 200 | 0.282 | 0.215 |
| 250 | 0.285 | 0.215 |

Figure 12: Loss Convergence plot showing Training Loss (blue line with circles) and Validation Loss (orange line with circles) over 250 training steps. The Training Loss starts at approximately 0.325 and decreases to about 0.285. The Validation Loss starts at approximately 0.22 and decreases slightly to about 0.215.

Fig. 12. Qwen model’s training and validation loss during LoRA fine-tuning

![Figure 13: Confusion Matrix for the Qwen model. The matrix is a 40x40 grid showing the relationship between true labels (y-axis) and predicted labels (x-axis). The diagonal line of bright green squares indicates correct predictions. Other colored squares represent misclassifications between different digit classes.](91be14371a97fb5ce9eeb29ae18d07c3_img.jpg)

Figure 13: Confusion Matrix for the Qwen model. The matrix is a 40x40 grid showing the relationship between true labels (y-axis) and predicted labels (x-axis). The diagonal line of bright green squares indicates correct predictions. Other colored squares represent misclassifications between different digit classes.

Fig. 13. Qwen model prediction matrix

### D. Runtime Comparison

TABLE I  
INFERENCE-STAGE RUNTIME COMPARISON ON SOCCERNET TEST TRACKLET 263 BETWEEN THE BASELINE PARSeq RECOGNITION PATH AND THE PROPOSED QWEN COLLAGE-BASED RECOGNITION PATH.

| Method | Input Type | # Inputs | Total Time (s) | Mean Time |
|-|-|-|-|-|
| Baseline PARSeq | individual crops | 635 | 9.05 | 14.24 ms/crop |
| Proposed Qwen | pre-built collages | 5 | 1.19 | 238.03 ms/collage |

We conducted an inference-stage runtime comparison on the same SoccerNet tracklet in a shared Google Colab A100 GPU environment to investigate runtime differences between the baseline and the suggested system. This experiment aimed to compare the recognition workload of our collage-based Qwen OCR path with the original crop-based PARSeq OCR path. This reflects our system’s overarching design objective where our approach substitutes OWLv2-based direct number localization with pose-estimation-based proxy localization prior to OCR, followed by recognition on a significantly reduced number of collage inputs. The measured runtime comparison only considers the OCR inference stage rather than the entire localization-and-recognition pipeline because OWLv2 itself was not timed. In this way, the experiment aims to demonstrate the practical recognition-time benefit of decreasing the number of recognition calls following upstream pipeline design simplification.

As shown in I, the results demonstrate a clear distinction in how computation is divided between the two recognition methods. For 635 crop-level forwards, PARSeq took an average of 14.24 ms per crop, totaling 9.05 s. Qwen processed five pre-built collages in a total of 1.19 seconds, averaging 238.03 milliseconds per collage. The collage-based formulation required far fewer total recognition calls, which led to a significantly reduced total inference time for that tracklet-level workload, even if each Qwen generation call was slower than a single PARSeq crop forward. Therefore, even though the timed comparison is restricted to PARSeq inference versus Qwen inference and excludes OWLv2 runtime, these results indicate that the proposed recognition architecture is faster in practice.

## VI. FUTURE WORK

### A. Data Cleaning

We noticed that there are many incorrect labels in the data set; an example is provided in Figure 14. In this case, the correct answer should be 38, even though our model’s prediction is also wrong (8) due to single digit cropping, the number is still reasonable and arguably partly correct. However, the ground truth for this sample is 1, which is a complete mislabel. The model training has reasonable resistance to small amount of noisy labels in the dataset, but the evaluation is not, these error labels made the eval accuracy lower than the model’s actual performance.

{9}------------------------------------------------

![Figure 14: A 3x5 grid of blurry images showing jersey numbers. Above the grid, the text 'GT: 1 | Pred: 8' is displayed in red, indicating a ground truth of 1 and a prediction of 8.](e7cb11f042fc58088dff4b6d9306845e_img.jpg)

Figure 14: A 3x5 grid of blurry images showing jersey numbers. Above the grid, the text 'GT: 1 | Pred: 8' is displayed in red, indicating a ground truth of 1 and a prediction of 8.

Fig. 14. An example of wrong label in the dataset.

### B. Reinforcement Learning

Reinforcement learning, specifically GRPO [16]–[18], has been shown to increase the ability of the model to reason. While our task is not traditionally reasoning-heavy, it does include reasoning parts like finding the right answer in noisy data. Since the reward of our task is simple (directly comparing with ground truth) a DeepSeekMath style reinforcement learning pipeline could be set up. The model output using chain-of-thought [19] before answer the question. However, this makes the latency of the output very high and is suitable only in high accuracy situations.

### C. Training Free Methods and Tools Augmented Agents

Our method trained a simple, small, and effective VLM (Qwen) to reorganize the numbers. However, with the rise of Large Vision Language Models (LVLM), the pipeline could be converted into a training free agentic pipeline. An LVLM, the larger variance of Qwen for example, could be feed directly with the collage. Then the LVLM directly answer the question about which number is in the image by themselves given a list of rules. The rules could include ignore single digit number when double digits present, or ignore obfuscating players and only focus on the main player. The LVLM could also receive a list of programmable tools to process the image for better visual understanding [20]. This results in a more portable and training-free pipeline. It also eliminates the need for local compute and can be deployed in many edge systems.

### D. Synthetic Data

One issue with the current system is the following. dataset is noisy and many tracklet has no useable groundtruth due to blurry images, one improvement we can make is high quality synthetic data. This can be done in two ways, rendering based or AI based. [21], [22] used Blender to render images

in replacement for the lack of real world data. We can use a similar approach to generate more data with hardcoded groundtruth. These data then could be used to train a more robust model.

## VII. CONCLUSION

This work presents a tracklet-based jersey number recognition pipeline that combines open-vocabulary detection, collage-based aggregation, and vision-language modeling. By shifting from frame-level recognition to a tracklet-level formulation, the approach aggregates information across multiple frames. This could help mitigate challenges such as occlusion, motion blur, and partial visibility. Real-ESRGAN improves the visual quality of low-resolution crops before detection, while OWLv2 enables tighter and more direct number localization. The collage representation, combined with a fine-tuned Qwen vision-language model, allows the system to integrate information across multiple frames without relying on handcrafted aggregation heuristics.

Beyond the specific pipeline design, this work explores an alternative formulation of jersey number recognition. Rather than relying on a single crop, the task is treated as reasoning over multiple imperfect observations distributed across a tracklet. This shifts the problem away from traditional OCR-style recognition toward aggregation and decision-making under uncertainty, where partial and noisy visual cues are combined to produce a final prediction. In this sense, the contribution lies not only in the system itself, but also in exploring how vision-language models can be applied to fragmented visual input.

At the same time, there are still some limitations to this approach. The system is computationally expensive, and its reliance on pretrained components can introduce failure cases under domain-specific conditions. In addition, dataset noise and annotation errors affect both training and evaluation. Supporting experiments on alternative legibility classifiers also provided useful design insights, although they were not incorporated into the final system. Overall, these results suggest that vision-language models with tracklet-level aggregation can be useful for jersey number recognition under noisy visual conditions.

## REFERENCES

- [1] M. Koshkina and J. H. Elder, "A general framework for jersey number recognition in sports video," in *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops*, pp. 3235–3244, June 2024.
- [2] B. Balaji, J. Bright, H. Prakash, Y. Chen, D. A. Claudi, and J. Zelek, "Jersey Number Recognition using Keyframe Identification from Low-Resolution Broadcast Videos," Sept. 2023. arXiv:2309.06285 [cs].
- [3] S. Gerke, K. Muller, and R. Schafer, "Soccer Jersey Number Recognition Using Convolutional Neural Networks," in *2015 IEEE International Conference on Computer Vision Workshop (ICCVW)*, (Santiago, Chile), pp. 734–741, IEEE, Dec. 2015.
- [4] K. Vats, W. McNally, P. Walters, D. A. Claudi, and J. S. Zelek, "Ice hockey player identification via transformers and weakly supervised learning," Apr. 2022. arXiv:2111.11535 [cs].
- [5] K. Vats, M. Fani, D. A. Claudi, and J. Zelek, "Multi-task learning for jersey number recognition in Ice Hockey," Aug. 2021. arXiv:2108.07848 [cs].

 Rest of paper (reference and Appendix) is removed.
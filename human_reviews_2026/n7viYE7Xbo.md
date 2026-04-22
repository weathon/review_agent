# BigMaQ: A Big Macaque Motion and Animation Dataset Bridging Image and 3D Pose Representations

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 4, 8, 2, 6

## Abstract
The recognition of dynamic and social behavior in animals is fundamental for advancing several areas of the life sciences, including ethology, ecology, medicine and neuroscience. Recent progress in deep learning has enabled an automated recognition of such behavior from video data. However, an accurate reconstruction of the three-dimensional (3D) pose and shape has not been integrated into this process. Especially for non-human primates, the animals phylogenetically closest to humans, mesh-based tracking efforts lag behind those for other species, leaving pose descriptions restricted to sparse keypoints that are unable to fully capture the richness of action dynamics. To address this gap, we introduce the $\textbf{Big Ma}$ca$\textbf{Q}$ue 3D Motion and Animation Dataset ($\texttt{BigMaQ}$), a large-scale dataset comprising more than 750 scenes of interacting rhesus macaques with detailed 3D pose descriptions of skeletal joint rotations. Recordings were obtained from 16 calibrated cameras and paired with action labels derived from a curated ethogram. Extending previous surface-based animal tracking methods, we construct subject-specific textured avatars by adapting a high-quality macaque template mesh to individual monkeys. This allows us to provide pose descriptions that are more accurate than previous state-of-the-art surface-based animal tracking methods. From the original dataset, we derive BigMaQ500, an action recognition benchmark that links surface-based pose vectors to single frames across multiple individual monkeys. By pairing features extracted from established image and video encoders with and without our pose descriptors, we demonstrate substantial improvements in mean average precision (mAP) when pose information is included. With these contributions, $\texttt{BigMaQ}$ establishes the first dataset that both integrates dynamic 3D pose-shape representations into the learning task of animal action recognition and provides a rich resource to advance the study of visual appearance, posture, and social interaction in non-human primates. The code and data are publicly available at [https://martinivis.github.io/BigMaQ/](https://martinivis.github.io/BigMaQ/).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present BigMac3D, a large dataset derived from hundreds of scenes of interacting rhesus macaques in a laboratory setting. Each scene is accompanied by 3D pose descriptions in the form of keypoints and textured surface meshes; per-view 2D keypoints and segmentation masks; and per-scene action labels. The authors then provide baseline action recognition results using a variety of pose and video descriptors, and demonstrate model improvements when high-quality pose descriptors are added to video descriptors.

### Strengths
BigMac3D presents a highly valuable resource to the behavioral quantification community. The dataset is very complete in terms of videos, keypoints, meshes, masks, etc., as well as the behavior annotations. This allows various types of methods developers to utilize this dataset.

The authors have also provided action recognition performance for a variety of baseline models, which is useful for methods developers looking to benchmark their models on this dataset.

### Weaknesses
The manuscript as written lacks clarity in various ways. The distinction between BigMac3D and BigMac500 isn't obvious. The types and amount of data related to each dataset is also not clear. The actual pipeline for collecting the raw data and creating the derived quantities (3D labels, meshes, etc.) is never clearly laid out. Addressing these issues would vastly improve the mansuscript.

The manuscript also lacks enough details about the quality of the dataset. The metrics in Tables 5+6 are a start but these numbers should, for example, be shared by behavior category. Example videos overlaid with keypoints and/or bounding boxes and/or segmentation masks would provide better intution for dataset quality. It is common to perform some kind of post-processing on the outputs of various models (detectors/pose estimators/segmenters) to clean up obvious errors; either this was not done or it was not described. These sorts of quality control steps are crucial before releasing such a large dataset to the community.

### Questions
Section A.4 describes the performance of the detection and pose estimation pipelines, but it is not clear from the reported metrics how less-than-perfect detections/poses influence mesh optimization. Furthermore, it is conceivable that these numbers will be highly dependent on the behavioral category, so it might be helpful to break them down this way. Section B.5 addresses this briefly, but I feel there should be much more space allocated to these issues.

Are there any post-processing steps used for the detection and pose estimation steps? For exmaple, multi-view (in)consistency across views could help flag detection errors.

There is no quantification of the segmentation masks from SAM2. Have the authors verified the performance of this model in any way? Has any post-processing been applied to clean up the segmentation masks? Presumably the predicted silhouettes would be helpful here? Is there a reason to include the SAM2 segmentation masks and not just the silhouettes output by the mesh-tracking stage?

It is not clear what the high-poly template model is used for.

How are skinning weights W chosen?

I am not super familiar with 3D mesh-tracking; it would be useful for that section to start with a few-sentence overview of the process before diving into specifics. For example, the appendix mentions this briefly, but it's not clear that the first step is to perform individual shape adaptation using the manual annotations, and then this information is used for the videos - this is not clear in the main text. Some other confusions I have:
- low- vs high-poly meshes, when are they used/not used
- SAM2 masks - are these used at all in the optimization process? If not, are they ever used to validate pieces of the optimization process?
- when are ground-truth keypoint annotations used in the pipeline, versus keypoints predicted by the pose estimation model?

The loss function in Eq. 2 contains a large number of hyperparameters. These values are given in Table 7 but there are no details on how these are chosen, or how sensitive the different hyperparameter values are.

Table 2: it is unclear what this actual metric is. The text says "IoU of mesh fits over time" - what is "ground truth" here? Is it IoU of meshes between time t and t+1? Table 2 legend says "first frame of the respective action", which seems contradictory. Please clarify.

L377: how is "reliable pose reconstruction" defined?

L411: the Kolesnikov reference should be Dosovitskiy

The conclusions from the action recognition results section are't entirely fair. The pose information is (presumably) sampled at 40 Hz, while the video information is sampled at 10 Hz. This confounds any direct comparison between the two. The more proper way to conduct this experiment to make the claims the authors want would be to downsample to pose data to 10 Hz or upsample the video data to 40 Hz. It is true that the former will lead to overall lower performance, and the latter to increased compute, and the hybrid results presented in the current submission could provide a useful approach for trading off accuracy and compute. But one of the aforementioned control experiments should be done.

Regarding the effects of pose representation, Blau et al. (https://arxiv.org/abs/2407.16727) looked at the incorporation of time into the pose representation and found a large performance improvement across multiple datasets. BigMac3D cool dataset to test these findings across a wider variety of spatial representations, if they are so inclined.

It's not actually clear what the difference between BigMac3D and BigMac500 are - BigMac3D has 750 "scenes", while BigMac500 has 8k "videos". Is one a subset of the other? Does BigMac500 not have segmentation masks, 2D keypoints, etc.? In general, a table with more exact details about what data is included in each dataset would help clarify this confusion.

Do the authors plan to publicly release the manual keypoint annotations? These could be very valuable for developers of 3D/multi-view pose estimation models.

It would be useful to provide train/test split information for the action recognition part so that new users can compare other models.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces BigMac3D, a large-scale dataset of over 750 multi-view video scenes capturing rhesus macaques performing diverse individual and social behaviors. Using 16 synchronized calibrated cameras, the authors reconstruct 3D surface-based pose and shape representations through a markerless motion-capture pipeline with skeletal rigs (115 joints) and subject-specific meshes. The dataset provides pose vectors, segmentation masks, keypoints, bounding boxes, and ethogram-aligned action labels, forming the first large-scale benchmark coupling 3D surface pose with action recognition in non-human primates. A derived benchmark, BigMac500, evaluates how these pose descriptors improve action classification performance when fused with various vision backbones (ResNet50, ViT, DINOv2, VideoPrism), achieving consistent mAP gains (≈+8–12%)

### Strengths
1.The dataset is unprecedented in scale and annotation richness for macaques, combining high-fidelity 3D surface tracking with ethogram-based behavioral labels across multi-individual interactions

2.The work builds on recent multi-view surface tracking advances (MAMMAL, AniMer+) but contributes practical improvements, such astemporal regularization, cropped differential rendering, and texture fitting, to enable efficient processing of thousands of sequences.

3.Comparisons against MAMMAL and AniMer+ confirm clear reconstruction gains (IoU = 0.84 vs 0.71/0.59). The action-recognition experiments further demonstrate that incorporating 3D pose vectors significantly boosts mAP across all tested encoders.

4.The dataset addresses a genuine need in neuroscience and ethology by bridging vision models with interpretable 3D motion cues, promoting reproducible behavioral analyses for NHP research.

### Weaknesses
1.The paper omits several critical details necessary for reproducibility, such as frame selection criteria, train/test split definitions, and subject-level separation (e.g., whether BigMac500 avoids identity leakage across splits). Although the authors describe a 70/10/20 split for the BigMac500 action recognition subset, this rule applies only to that benchmark, not the full BigMac3D dataset. There is no explicit statement on whether subjects are disjoint across splits. Furthermore, while the dataset uses a 16-camera synchronized capture setup, no visualization of camera IDs, baseline distances, or view diversity is provided in any figure or table, making it difficult to assess spatial coverage or view redundancy. These omissions weaken the dataset’s clarity and reproducibility.

2.While the authors claim “high-quality surface reconstruction,” the paper provides no quantitative 3D validation, such as per-joint reconstruction error or ground-truth comparison. Qualitative figures (e.g., Fig. 3) indeed show visually strong fits, but the absence of uncertainty estimates, human-verified subsets, or error histograms leaves the actual accuracy unquantified. This weakens confidence in downstream claims about 3D pose reliability and consistency across subjects.

3.Related to the first issue, the lack of clarity on subject-level isolation across training and testing sets raises potential concerns about identity memorization in the reported benchmarks. Explicitly ensuring disjoint subjects would help establish a fair evaluation protocol for behavior recognition tasks.

### Questions
Could the authors provide more details on how the dataset is organized and validated? In particular, it would be helpful to clarify how the training and testing splits are defined for BigMac3D and BigMac500, and whether subjects are disjoint across splits to avoid identity memorization. Additionally, more explanation on how 3D pose accuracy and annotation confidence were assessed would strengthen the paper’s reliability. Finally, reporting key dataset statistics such as frame sampling rules, camera ID distributions, and sequence lengths would improve the dataset’s transparency and reproducibility.

### Soundness
3

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents BigMac3D, a large-scale 3D dataset of macaques that bridges visual appearance and 3D pose representations for action recognition. The dataset comprises over 750 multi-view scenes recorded with 16 synchronized cameras and annotated with 3D skeletal joint rotations, segmentation masks, 2D keypoints, and ethogram-aligned behavioral labels.
To generate these annotations, the authors design a markerless motion capture pipeline that integrates surface-based mesh tracking with subject-specific textured avatars, enabling precise reconstruction of individual body shapes and motions. They further introduce BigMac500, a curated benchmark subset for evaluating pose-informed action recognition, demonstrating that incorporating 3D pose descriptors leads to significant gains in mean average precision (mAP) across diverse visual backbones.

### Strengths
### **Strengths**

* The paper introduces a large-scale dataset that integrates detailed 3D pose–shape representations with action labels for macaques, addressing an important gap in current non-human primate research.
* The proposed annotation and reconstruction pipeline is innovative, combining markerless motion capture, subject-specific mesh tracking, and photometric texturing in a scalable way.
* The BigMac500 benchmark is a valuable contribution, demonstrating quantifiable improvements in action recognition performance when incorporating 3D pose descriptors.
* The annotation process **appears** well designed and yields annotations of good quality.
* The paper is simple and well written.
* The research problem is both interesting and challenging.
* The scale and diversity of the dataset is appealing.

### Weaknesses
### **Main Weaknesses**

The paper currently lacks sufficient detail about the dataset annotation pipeline. A clear, visual overview of the entire annotation process would be highly valuable—ideally presented as a dedicated figure (either in the main paper or, at minimum, in the appendix).

Secondly, the loss functions introduced for optimizing the annotations are not supported by ablation studies. Without these, it remains unclear which losses meaningfully contribute to the final annotation quality, and to what extent.

Moreover, there is **no quantitative evaluation of the annotation quality** beyond Table 2, which only reports IoU scores. This metric assesses 2D mask alignment but provides limited insight into the accuracy of the 3D surface models. It would be important to include additional metrics that reflect 3D fidelity and realism—such as *PSNR*, or *KL divergence*—as well as a measure of temporal consistency across frames. Since multiple losses are designed to optimize different parameters, it would be critical to demonstrate, both quantitatively and qualitatively, how these losses affect the reliability of the learned representations.

Finally, while the reported IoU of 0.844 is promising, it requires further clarification. Where do the remaining percentage points of error originate? A visualization comparing predicted and pseudo-ground-truth masks would help interpret this discrepancy—distinguishing between errors due to poor mesh fitting (undesirable), inaccurate pose estimation (undesirable), imperfect 2D masks (potentially acceptable), or minor border inaccuracies (acceptable).

### **Other Weaknesses**

* The paper provides no information about computational requirements. Details on runtime, GPU hardware, and annotation throughput should be included.
* The dataset was collected in controlled laboratory conditions, which may limit generalization to in-the-wild scenarios.
* The pose baseline used for prediction is not described. The paper should clarify the model architecture, input format, and training procedure.
* Loss ablations should be provided wherever feasible .

### Questions
### **Questions for the authors**

* I am not fully clear on what using “Real” rather than “Synthetic” 3D data concretely enables. Could you clarify the specific advantages it brings for model quality, bias reduction, or downstream generalization?
* How were the action labels produced in practice? Please describe who annotated them, the protocol, the ethogram mapping procedure, inter-annotator agreement, and any quality control. I could not find these details in the paper or the cited appendix.
* Can you provide evidence that the benchmark gains transfer to in-the-wild data? For example, a cross-domain test, few-shot adaptation, or evaluation on an external macaque or primate dataset.
* How well does the model personalize to individual monkeys? Concretely, how morphable is the 3D template, and what quantitative measures do you have on shape fitting accuracy across individuals?
* Do you quantify annotation uncertainty for the intermediate labels (segmentation, keypoints, identities) and analyze how these errors propagate to the mesh fitting and final pose vectors?
* How robust is the method to partial occlusion or missing views? Please report performance as a function of visible views, occlusion level, and view dropout, and describe any mechanisms that was used for handling this.

### **Summary comment**

The paper is strong and promising, but for a dataset and pipeline of this complexity, the current qualitative evidence is not sufficient. A set of targeted quantitative analyses would make the contribution much more convincing: ablations for each loss term, uncertainty propagation from 2D annotations to 3D fits, view-drop robustness, transfer to in-the-wild scenarios, and individual-specific morphability metrics. If provided, these results could substantially strengthen the work and its impact.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents BigMac3D, a large multi-view video dataset of rhesus macaques with dense surface-based 3D pose/shape reconstructions and action labels (an ethogram). The recordings come from 16 calibrated cameras in a neuroscience laboratory and cover >750 action scenes and a derived action-recognition benchmark (BigMac500) of $~\sim$8k labeled videos. The authors build subject-specific textured avatars by adapting a high-quality macaque template mesh and introduce processing/optimization improvements (symmetric time loss, cropped differential rendering, integrated texture) to make large-scale surface reconstruction tractable.

### Strengths
Impactful dataset: fills an important gap for non-human primate research by linking dense 3D surface reconstructions with behavior labels.

Practical pipeline: believable and reproducible-in-principle mesh-fitting and rendering pipeline that scales to hundreds of scenes.

Demonstrated utility: empirical evidence that 3D pose/shape descriptors can improve action recognition across multiple visual backbones.

Rich annotation set: identities, segmentation masks, 2D keypoints, calibrated views, per-frame poses and action labels.

### Weaknesses
Ethical / animal welfare documentation (major). The recordings come from a neuroscientific laboratory and involve captive macaques. I cannot find any explicit statement of oversight, animal-use committee (e.g., IACUC or institutional equivalent) approval, or details of welfare protocols. The paper says recordings were “not induced by humans except feeding,” but ethical approval and animal care protocols must be stated explicitly. 

Statistical significance and variance. Report standard deviations / confidence intervals and number of seeds for all recognition results (Tables 3-4) and any other ML experiments. If training is deterministic (unlikely), explain why; otherwise run multiple seeds and report variability. Also provide basic compute footprints (GPU types, training time per model, FLOPs if possible) so the community can compare methods fairly.

Annotation reliability. How were action labels assigned? How many annotators per clip? Multi-label or single-label? What was the inter-annotator agreement (kappa / % agreement)? Actions like “aggression,” “dominance display,” or “anxiety” can be subjective—please document.

Dataset bias & generalization. The dataset contains eight male macaques in one lab. Discuss how sex, age, enclosure, and captive vs. wild behavior may affect generality. Are there plans to extend diversity (females, juveniles, different environments)?

Privacy / safety considerations. While this is animal data, the authors should discuss potential misuse (e.g., tracking in the wild or exploiting behavior data) and state any restrictions on dataset use if applicable.

### Questions
Refer to questions

### Soundness
2

### Presentation
3

### Contribution
2

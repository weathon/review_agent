# Learning to focus on target for weakly supervised visual grounding

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3

## Abstract
Visual grounding is a crucial task for connecting visual and language descriptions by identifying target objects based on language entities. However, fully supervised methods require extensive annotations, which can be challenging and time-consuming. Weakly supervised visual grounding, which only relies on image-sentence association without object-level annotations, offers a promising solution. Previous approaches have mainly focused on finding the relationship between detected candidates, without considering improving object localization. In this work, we propose a novel method that leverages Grad-CAM to help the model identify precise objects. Specifically, we introduce a CAM encoder that exploits Grad-CAM information and a new loss function, attention mining loss, to guide the Grad-CAM feature to focus on the entire object. We also use an architecture that combines CNN and transformer, and a multi-modality fusion module to aggregate visual features, language features, and CAM features. Our proposed approach achieves state-of-the-art results on several datasets, demonstrating its effectiveness in different scenes. Ablation studies further confirm the benefits of our architecture.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a method for the task of weakly-supervised visual grounding. The method consists of four main parts: visual encoder, language encoder, CAM encoder, and multi-modality fusion part. The proposed method is trained with a new proposed loss fucntion. The experimental results on several mainstream datasets show the good performance compared with state-of-the-art methods.

### Strengths
The performance of the proposed method looks good on several Ref datasets compared with the others. It shows the effect of the weally-supervised on this challenging vision-language task. The paper also verifies the effect of the grad-cam features that can play an important role in the precise boundbing box prediction.

### Weaknesses
There are several main weakness of this paper, and here follows the details:

1. The novelty of the proposed method. All the four main parts and their architecture are not new in the domian of vision-language. This kin of easy combination does not make a strong techinical contribution to this task.

2. I am really interested in how to make use of the prior attention maps from Grad-CAM. Unfourtunately, I did not find the detailed information  about it. This could be the biggest stregth of this paper.

3. A lot of details are missing in this paper, for example, which version of BERT did the authors adopt? How to address the difference of the feature dimensions when fusing them? 

4. Figure 5, Here I do not understand that with a regression predictor, how can the authors relate the predicted ones with the natural language words or nouns? More common cases could be multiple bounding boxes, how can the authors filter that and correspond that?

### Questions
1. The authors mention "focus on the entire object". How did they do that? And how to measure the difference of entire object and partial attention maps in visual grounding task?

2. The authors pointed out the weakness of transformer-based methods, and claimed the proposed method could addrees that on Page 2. Why and how to measure that?

3. I suggest to introduce Grad-CAM method and the prior works based on it on Page 3.

4. The content of the caption is repeated.

5. Can the authors describe the details of the loss compared with that in GAIN?

6. Have the authors done the ablation study about the loss function? The hyper-parameters are very important in this kind of training procedure.

7. The experimental results did not show quite big difference between various versions. Which part do the authors think the proposed fusion method really helps?

### Soundness
2 fair

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work proposes a method for weakly supervised visual grounding. They design a model that consists of an image encoder, a text encoder, a CAM encoder, and a Multi-Modality Fusion Module to fuse features from different encoders and predict bounding boxes for input text phrases. The novel part of the model is the CAM encoder, in which they first obtain a Grad-CAM heatmap from the input image, text and the CNN backbone, and use the heatmap as the input for the CAM encoder. They use the CAM encoder to calculate the attention mining loss to help the CAM features focus on the whole object mentioned in the text. They also adopt a self-taught regression
loss and a phrase reconstruction loss proposed by Liu et al. to train the model. They compare with some previous methods on both weakly supervised visual grounding and fully supervised visual grounding on several benchmarks.

### Strengths
First, the experiments are comprehensive. They validate their methods on multiple benchmarks including RefCOCO, RefCOCO+, RefCOCOg, Flickr30, and ReferItGame. 
Second, they conduct a bunch of experiments to show the effectiveness of each component in the model design, including the feature selection for the multi-modality fusion module and for the image encoder.

### Weaknesses
(1) Missing baselines for comparison. In this work, the author also uses an object detector for generating proposals, which is a usual setting for weakly supervised visual grounding. However, in the experimental section, the authors do not discuss or compare with multiple previous proposed methods such as [1-4]. If the authors add the results from these works, they will not achieve state-of-the-art results for at least Flickr30k and RefCOCO+. 

(2) The authors claim "... Nevertheless, no previous attempts have been made to integrate Grad-CAM with existing weakly supervised visual grounding methods ...", and "... Although many methods have been proposed to improve the performance of weakly supervised visual grounding, none have made use of Grad-CAM, which is commonly used in weakly supervised training ...". However, in [4], they also use Grad-CAM to help the visual grounding. 

(3) Missing discussion of previous works in weakly supervised visual grounding. In this paper, the authors still use object detectors to do the weakly supervised visual grounding. Unfortunately, training an object detector requires bounding box annotations (in this case, bounding box annotations in the VG dataset). Therefore, it is not exactly "weakly supervised" since the model still requires bounding box annotations to train the object detector. There exist works [5,6] that only use images and text for weakly supervised visual grounding, but the author did not discuss them in Section 2. 

References:

[1] Dou, Zi-Yi, and Nanyun Peng. "Improving pre-trained vision-and-language embeddings for phrase grounding." Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing. 2021.

[2] Wang, Qinxin, et al. "MAF: Multimodal Alignment Framework for Weakly-Supervised Phrase Grounding." Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing (EMNLP). 2020.

[3] Chen, Keqin, et al. "Contrastive Learning with Expectation-Maximization for Weakly Supervised Phrase Grounding." Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing. 2022.

[4] Yang, Ziyan, et al. "Improving Visual Grounding by Encouraging Consistent Gradient-based Explanations." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

[5] Arbelle, Assaf, et al. "Detector-free weakly supervised grounding by separation." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[6] Shaharabany, Tal, and Lior Wolf. "Similarity Maps for Self-Training Weakly-Supervised Phrase Grounding." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
(1) What is the object detector used in the method?

(2) Please polish the figures in the paper. For example, Figure 4 is too small, making the text in the figure unclear. 

(3) For the qualitative results, can you provide the results for the same samples from other methods as well? That will make the qualitative results more impressive.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

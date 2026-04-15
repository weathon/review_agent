# Explaining Vision-Language Similarities in Dual Encoders with Feature-Pair Attributions

- Decision: Reject
- Scores: 6, 5, 6, 5

## Abstract
Dual encoder architectures like CLIP models map two types of inputs into a shared embedding space and learn similarities between them.
  However, it is not understood how such models compare the two inputs.
  We first derive a method to attribute predictions of any differentiable dual encoder onto feature-pair interactions between its inputs. 
  Second, we apply our method to CLIP models and show that they learn fine-grained correspondences between parts of captions and regions in images. They match objects across input modes and also account for mismatches. However, this visual-linguistic grounding ability heavily varies between object classes, depends on the training data distribution, and largely improves upon in-domain training.
  Using our method we can identify individual failure cases and knowledge gaps about specific object classes.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper extends the Integrated Gradients algorithm to explain similarity in CLIP models which consist of two vision-language encoders, via attributing feature pairs (language-vision attribution). This allows capturing of feature interactions and correspondances, rather than having 2 separate attribution entities that are disconnected. Experiments are conducted on a variety of CLIP models from OpenAI and OpenCLIP.

### Strengths
- The authors tackle an important topic of interpretation of feature interactions, which is not that well-explored in the literature
- Wide range of CLIP models considered in experiments
- Nice extension of the work "An Attribution Method for Siamese Encoders to CLIP" to the vision-language setting.

### Weaknesses
- [W1] There is an abundance of related work that the authors miss. [R1, R2, R3] are papers that are doing exactly the same thing. R1 proposes several baseline methods and evaluation metrics. R3 is also applied to CLIP and shows object-word interactions. The authors did not mention these works, their drawbacks and how their method is better, and did not compare with them. The authors are proposing a method without comparing it to existing methods in the literature. Comparison with baselines (and mentioning how the proposed method is different and what advantages it offers) is an integral part of any research work, which is missing. 

- [W2] Related to W1, the authors do not mention neither prove how methods which provide individual feature attributions (e.g., Grad-CAM, Integrated Gradients, LIME...etc) are bad and do not work, and that a new formulation of feature interactions is needed. Simply stating this statement without proving it, is not valid. 

- [W3] Poor evaluation: all I see is two graphs in Figure 5 (table 1 is an ablation experiments, not main results). There is no wide enough evaluation metrics and tasks. The only evaluation in Figure 5 is based on bounding box overlaps. However, it is well-agreed upon in XAI that bounding boxes are a very bad way of evaluating explanations, because they assume the model reasoning process aligns with the annotations.  If a model is biased and detects a "hand" rather than a "dumbell", it will be penalized (because the annotated box is for the dumbell), while it should not. Similarly, assume the model can identify a dog solely by its "tail". In this scenario, the model focuses only on physical features of the tail, and disregards other features like the dog's body, fur, or face. A good explanation in this case would highlight the tail. But, the overlap between the bounding box of that explanation (the dog's tail) and the overall bounding box of the dog would be small and penalised (while it should not). These attribution methods act as a way to perform applications such as zero-shot object detection or segmentation (e.g., [R4]), but not as ways to evaluate interpretations. 

- [W4] How did the authors arrive to Eq.2? How is is formulated? Based on what? It suddenly appears in this form. Did the authors create this starting point? If so, based on what?

Minor (do not affect my decision):
- The authors mention: "One may expect CLIP models must learn object correspondence between vision and language modes. But to our knowledge, our evaluation is the first piece of evidence indicating that this is actually the case". This is not surprising at all. None of the various applications of CLIP would actually work if this was not the case. 
- Not sure what is (top: selections in yellow, bottom: saliencies as above) in Figure 2. Are yellow and red supposed to be corresponding? Or is red the negative? 

References\
[R1] Visualizing and Understanding Contrastive Learning, TIP 2023\
[R2] Visualizing deep similarity networks, WACV 2019\
[R3] Model-Agnostic Visual Explanations via Approximate Bilinear Models, ICIP 2023\
[R4] Interpreting CLIP's Image Representation via Text-Based Decomposition, ICLR 2024\

### Questions
In general, I feel this paper is not ready for ICLR. The weaknesses greatly outweighs the strength, Therefore, my decision will be, sadly enough, to reject this paper. W1, W2, W3 and W4 are major issues which should be addressed.

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
This paper is mainly focused on how dual encoder architectures map two types of inputs into one shared embedding space. To this end, authors propose to attribute predictions of any differentiable dual encoder onto feature-pair interactions between inputs. The proposed method shows that CLIP models learn fine-grained correspondences between parts of captions and regions in images. In-domain finetuning can largely improve the visual-linguistic grounding ability.

### Strengths
1. The paper is well written and easy to follow.
2. The authors provide extensive illustrations to demonstrate the visual-linguistic alignment results.

### Weaknesses
1. While authors demonstrate the improvements by applying the proposed method to CLIP modes, the results seem to be straightforward: (1) visual-linguistic grounding ability heavily varies between object classes and the training data distribution; (2) in-domain finetuning can largely improve such ability.

2. Regarding the results in Table 1, there is one missing baseline which is directly finetuning CLIP models without the proposed method. Such ablated experiment can further show the effectiveness of the proposed method.

3. Why are there some missing results for the HNC dataset in Table 1?

### Questions
See the above weakness. 

1. The main concern is that there is one missing baseline in Table 1. This baseline should finetune CLIP models without the proposed method to ablate the proposed method.

2. It seems straightforward to get the conclusions that in-domain finetuning can improve the visual-linguistic ability. It is a little bit unclear about the main contributions of this paper.

3. Have you ever compared the T2I and I2T retrieval results w/ and w/o finetuning? Does better visual grounding lead to better retrieval ability?

### Soundness
2

### Presentation
2

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
This paper presents a method to interpret CLIP models' cross-modal alignment mechanisms by attributing similarity predictions to specific feature-pair interactions across input modalities (image-text, text-text, image-image). The proposed method leverages integrated gradients to match parts of captions with corresponding regions in images. The authors also show that the model’s grounding abilities vary by object class and are sensitive to the training data distribution, with in-domain fine-tuning improving performance. The paper includes evaluations on datasets with object bounding-box annotations (3.5k image-caption pairs from COCO, 8k pairs from Flickr30k, and 500 pairs from HNC).

### Strengths
- This paper is easy to read, the proposed algorithm/methodology is sound, and the technical details are well explained (including a general description for code implementation in Section 7).

- This paper is an extension of previous work for language-only Siamese models in NLP (Möller et al., 2023; 2024) -- it provides a principled implementation for assessing the visual-linguistic grounding abilities of CLIP models, showing that the proposed approach in [1, 2] generalizes to vision and language models (particularly, image-text and image-image matching).

- The proposed generalization of integrated gradients for object attribution is simple yet effective, and seems to work well for CLIP models.


[1] Lucas Möller, Dmitry Nikolaev, and Sebastian Padó. An attribution method for siamese encoders. In Proceedings of EMNLP, Singapore, 2023.

[2] Lucas Möller, Dmitry Nikolaev, and Sebastian Padó. Approximate attributions for off-the-shelf
Siamese transformers. In Yvette Graham and Matthew Purver (eds.), Proceedings of the 18th
Conference of the European Chapter of the Association for Computational Linguistics (Volume 1:
Long Papers), pp. 2059–2071, St. Julian’s, Malta, March 2024. Association for Computational
Linguistics.

### Weaknesses
*"One may expect CLIP models must learn object correspondence between
vision and language modes. But to our knowledge, our evaluation is the first piece of evidence
indicating that this is actually the case."* -- this is a strong claim that does not hold. A large number of prior works has systematically explored gradient-based methods that uncover the grounding capabilities of CLIP (and other visual-language models) [3, 4, 5, 6]. 

- Limited evaluation: gradient-based explanations are typically evaluated in pointing game accuracy, segmentation masks, object detection and qualitative comparison against prior work. This work only compares the grounding ability in bounding-box evaluation. It is unclear how the proposed approach fares against prior relevant work.

- Due to limited benchmarks and testbeds, it is difficult to determine if the proposed method is competitive against prior work, and to what extent it provides significant new contributions.

[3] Chenyang ZHAO, Kun Wang, Xingyu Zeng, Rui Zhao, & Antoni B. Chan (2024). Gradient-based Visual Explanation for Transformer-based CLIP. In Forty-first International Conference on Machine Learning.

[4] Yossi Gandelsman, Alexei A Efros, & Jacob Steinhardt (2024). Interpreting CLIP's Image Representation via Text-Based Decomposition. In The Twelfth International Conference on Learning Representations.

[5] Weĳie Tu, Weijian Deng, & Tom Gedeon (2023). A Closer Look at the Robustness of Contrastive Language-Image Pre-Training (CLIP). In Thirty-seventh Conference on Neural Information Processing Systems.

[6] Bousselham, Walid, et al. "LeGrad: An Explainability Method for Vision Transformers via Feature Formation Sensitivity." arXiv preprint arXiv:2404.03214 (2024).

### Questions
- As an extension of previous work, the contribution seems incremental due to the limited experimental analysis. Is it possible to show experimental results with other VLM architectures?

- Results in Figure 5 are difficult to parse. Is it possible to separate the results into different tables per model (to show in the Appendix)?

_____

Not questions to answer, but suggestions that don't need to be addressed: 
- What are the results after finetuning with out-of-domain data? -- this might give a better picture and analysis of the models' performance and explainable approach

- In Related Work (Sec. 2), there is a rather extensive and detailed explanation of vision-language models and their training objectives, however,  this work only focuses on CLIP models -- it might be interesting to correlate how different learning objectives/models correlate with the proposed post-hoc evaluation, and how they impact the grounding/explanation capabilities exposed by prior work and the proposed approach.

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
3

### Summary
This paper investigates the mechanisms by which dual encoder architectures, such as CLIP models, compare two types of inputs (e.g., text and images) by mapping them into a shared embedding space and learning their similarities. To address the lack of understanding of how these models compare inputs, the authors present two main contributions:

Derivation of Feature-Pair Attribution Method: The authors derive a method to attribute the predictions of any differentiable dual encoder onto the feature-pair interactions between its inputs. This method can explain interactions between inputs of any differentiable dual encoder model without requiring modifications to the trained model.
Application to CLIP Models: The authors apply their attribution method to CLIP-type models and demonstrate that these models can learn fine-grained correspondences between parts of captions and regions in images. They show that the models can match objects across input modes and account for mismatches. The visual-linguistic grounding ability of these models varies significantly between object classes, depends on the training data distribution, and improves with in-domain training.

The paper includes various experiments to validate the proposed method, such as:
Evaluating object bounding-box attributions to systematically assess the visual-linguistic grounding abilities of dual encoders.
Analyzing attributions to other objects to identify negative attributions for mismatched objects.
Creating and evaluating hard negative captions to test the model's response to errors in captions.

### Strengths
1. The paper is well-written and clearly structured.

2. The mathematical derivations and equations are clearly presented (But I did't check carefully)

3. The proposed feature-pair attribution method has the potential to significantly advance the understanding of how dual encoder models, particularly vision-language models, compare inputs and make predictions. This understanding is crucial for improving model interpretability and trustworthiness.

### Weaknesses
1. The authors acknowledge that their feature-pair attribution method is an approximation and may not fully capture the exact contributions of feature interactions. This approximation introduces potential inaccuracies in the interpretation of attributions.

2. The experiments and evaluations are primarily focused on CLIP models and specific datasets (e.g., COCO, Flickr30k, HNC). Conducting experiments on a broader range of dual-encoder models and datasets from different domains (e.g., medical imaging, audio-visual data) could demonstrate the generalizability and versatility of the method.

3. While the paper introduces a promising analysis method for attributing predictions of dual encoder models to feature-pair interactions, it lacks a straightforward way to quantify the performance of existing models using this method.
The absence of a standardized metric, similar to FID in image generation or BLEU [1] in machine translation, limits the broader applicability and impact of the method. Having such a metric could significantly enhance the influence and usability of the proposed method.

4. The paper primarily focuses on analyzing and explaining existing phenomena within CLIP models rather than providing solutions or improvements to the training process. While this analysis is valuable, it leaves several important aspects of CLIP training unexplained.
For example, the paper does not address why CLIP models require an extremely large batch size during training, which is a critical aspect of their performance and efficiency.

[2] [Bleu: a Method for Automatic Evaluation of Machine Translation](https://aclanthology.org/P02-1040) (Papineni et al., ACL 2002)

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
2

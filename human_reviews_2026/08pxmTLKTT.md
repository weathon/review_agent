# SmartSAM: Segment Ambiguous Objects like Smart Annotators

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 4, 4, 6

## Abstract
Segment Anything Model (SAM) often encounters ambiguity in interactive segmentation, where insufficient user interaction leads to inaccurate segmentation of the target object. Existing approaches primarily address ambiguity through repeated human-model interactions, which are time-consuming due to the inherent latency of human responses. To reduce human efforts, we propose a novel interactive segmentation framework that leverages the model’s inherent capabilities to effectively segment ambiguous objects.
Our key idea is to create an annotator-like agent to interact with the model. The resulting SmartSAM method mimics intelligent human annotators, resolving ambiguity with a single click and one reference instance. The agent generates multiple prompts around the initial click to simulate diverse annotator behaviors and refines the output masks by iteratively adding click chains in uncertain regions, thereby producing a set of candidate masks. Finally, the agent selects the mask that most closely aligns with the user’s intent, as indicated by the reference instance. Furthermore, we formalize the agent’s behavior as a fuzzy regression problem by quantifying ambiguity using fuzzy entropy. We demonstrate that our agent yields lower entropy than traditional methods, and we establish robustness and sufficiency theorems to ensure effective, human-like decision-making within a bounded range of actions. We evaluate our approach on multiple segmentation benchmarks and demonstrate its superiority over state-of-the-art methods.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes to better address the problem of object ambiguity in interactive segmentation (IS) models with SmartSAM method. To achieve it, an agent generates a few branches with interactions (positive/negative click or bbox), considering the first user interaction, to produce candidate masks, then it compare every candidate with a reference object using cosine similarity and choose the most similar one. Using fuzzy statistics the paper demonstrates that its method achieved lower fuzzy entropy than traditional SAM. Finally, it demonstrates improved performance on DAVIS, PartImageNet and Amb-Occ (firstly presented in the paper) datasets of IS models with SmartSAM usage.

### Strengths
1) The method is training free with small amount of extra computation, due to the amortized inference of SAM-like models, when encoder and prompt are processed separately. These facts make the method quite valuable for real-world applications.
2) The authors provide proof of the efficiency of their method in fuzzy entropy, considering a few assumptions.
3) The method helps to achieve better qualitative results on the proposed dataset.
4) New proposed dataset, specially proposed for such task.
5) Ablations on usage of Gamma distribution to Normal, sampler choice, encoder.

### Weaknesses
1) Arguable that it has a lot of practical value in a wide meaning, because for every class of segmenting objects we need manually choose reference. In my opinion it could take much more time than just labeling with more clicks, e.g. you need to label different objects on image, for every object you need to find reference and choose it (even if all possible references will be in the interface of the labeling app it could take a sensitive amount of time to find needed one), then make a click (assume that we label it with just 1 click successfully), OR making 2 or 3 clicks without searching for reference, considering that we have amortized inference and segmentation is quite fast, the second method looks much faster for annotation of different classes.

Of course if there is only 1 class that must be annotated then this way should be faster.

2) As I understand for click simulation on DAVIS and PartImageNet you used clicking in the center of the object. But in [1, 2] there were demonstrated that people do not click in the center of object, also in [3] it was written that clicking in the center strategy leads to overfitting for NoC metrics that you also use.

3) There is no human study. That is why it is hard to estimate real-world applicability.

4) Comparison of the IS models with vs without SmartSAM is not fully fair, because SmartSam proposes more candidates and gets more information with reference.

5) Also comparison with [4] and [5] is not fully fair, because these methods get as input only reference, while SmartSAM takes first initial click in addition.

6) Also, compare the method with SAM is not fully fair, because SAM generates only 3 masks, while SmartSAM could generate many more.

A few typos: It is written sometimes HQSAM and sometimes HQ-SAM. It is written mIoU@1, Ratio@k and just NoC in Evaluation Metrics subsection, please, be consistent and write NoC@k.

[1] - TETRIS: Towards Exploring the Robustness of Interactive Segmentation (AAAI 2024)

[2] - RClicks: Realistic Click Simulation for Benchmarking Interactive Segmentation (NeurIPS 2024)

[3] - Reviving Iterative Training with Mask Guidance for Interactive Segmentation (ICIP 2022)

[4] - Bridge the Points: Graph-based Few-shot Segment Anything Semantically (NeurIPS 2024)

[5] - Matcher: Segment Anything with One Shot Using All-Purpose Feature Matching (ICLR 2024)

### Questions
1) Provide, please, more details about Table 6, when there is a column with NCC, PIS is absent, that is why it is hard to understand how the first clicks where achieved.

2) To close some gaps with real-world usage you can use [1], where it was proposed a dataset with real user clicks (including real user clicks for DAVIS). 

3) Also, please, explain how you suggest to use this method in real-world application with multi-class segmentation scenarios to improve the speed of annotators.

4) It would be valuable to use at least real-users first clicks in your tests, much better to use clickability model from [1], but only if you have enough time.

5) It is very interesting if additional finetuning of SAM on the ambiguous objects data could improve its performance.

The idea of the paper is quite interesting and after clarification of my doubts I will reconsider my score.

[1] - RClicks: Realistic Click Simulation for Benchmarking Interactive Segmentation (NeurIPS 2024)

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The Segment Anything Model (SAM) often encounters ambiguity in interactive segmentation, especially during the initial interaction. To reduce the need for extensive human input, the authors propose SmartSAM, a method aimed at improving segmentation accuracy. The key idea is to generate a diverse set of prompts around the initial user prompt, producing multiple candidate masks. The most appropriate mask is then selected by measuring feature similarity to the reference, computed using DINOv2 embeddings. Experiments on three benchmark datasets demonstrate the effectiveness of the proposed method.

### Strengths
The proposed method is training-free and can be plugged in on the fly during inference on top of different interactive segmentation models: SAM, HQ-SAM, HRSAM, FocSAM etc

The author also provides some theoretical analysis for the SmartSAM disambiguity, which demonstrates that the fuzzified candidate set exhibits less uncertainty.

### Weaknesses
The most significant weakness of this paper lies in its reliance on external exemplar images and corresponding masks. Comparing the proposed method mainly against the base SAM model is therefore not a fair evaluation, as the proposed method benefits from additional example pairs. The author provides some comparison with personalized segmentation methods, such as PerSAM and Matcher, where example image-mask pairs are leveraged to adapt SAM. However, the experiments are not comprehensive; it is only on the Davis dataset, while other widely used benchmarks in one-shot/ few-shot or personalized segmentation are not included.  Besides, the performance reported in Table 2 seems to be much lower than that in the PerSAM paper. I understand that the Davis in this paper may be different than the DAVIS 2017 val dataset. Still, given such a huge difference, I would like the author to give more details. 

While the paper claims to introduce a segmentation agent to guide SAM, the actual mechanism is a set of straightforward rule-based procedures. The core idea—comparing feature similarity between candidate masks and a reference image-mask pair using mask-pooled DINO features—has already been explored in prior works. Moreover, such similarity-based methods may struggle to capture fine-grained semantic distinctions between regions. The authors should provide visual examples to better demonstrate both the strengths and limitations of their approach.

### Questions
See above

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper considers a task of interactive object segmentation, using user prompts and reference (text- or visual-based). An agent-based method is proposed, which is training-free, and can be used on top of any interactive segmentation methods. The method has been evaluated using several variants of SAM-based methods, and demonstrated significant improvement in accuracy over baselines.

### Strengths
1) The proposed agent-based method is training-free, and can be based on any SAM-like segmentation model.
2) The proposed method leads to significant segmentation accuracy, compared to basedlines methods (however at X costs, due to several hypothesis, tested in parallel)
3) Authors has theoretically validated the proposed method, formulating several theorems to prove it.

### Weaknesses
1) Limited validation - missing benchmarks COCO-20i and PASCAL-5i. 
The idea of using text or visual references is not new, and several methods has been proposed to implement it (e.g. ProSAM https://arxiv.org/abs/2506.21835, VLP-SAM https://github.com/kosukesakurai1/VLP-SAM, VRP SAM https://openaccess.thecvf.com/content/CVPR2024/papers/Sun_VRP-SAM_SAM_with_Visual_Reference_Prompt_CVPR_2024_paper.pdf). However, they all tested on COCO-20i and PASCAL-5i. The proposed SmartSAM hasn't been tested on these benchmarks. So we cannot assess the accuracy of the proposed method relative to the competitor's ideas. 

2) Limited technical complexity - the idea is to basically sample several hypothesis by sampling virtual clicks ("inner interactions") in low-certainty areas, and select the best one by comparing features to visual/text prompt. The method is very simple and rather straight forward. Very "brute force"

3) Limited novelty - the usage of text/visual reference is very limited, only in selection of best hypothesis. In theory it could and should be used for prediction of "inner clicks", to better guide the selection of them.

### Questions
1) Why COCO-20i and PASCAL-5i hasn't been used? Why no comparison with sota reference-based methods?

Small issues:
* In the intto it is stated that "039 A key issue with these methods is ambiguous predictions caused by insufficient interactions, where
models often misinterpret the user’s intent, leading to undesired segmentation masks.". However as it is seen from futher description the key problem is that user intent is unknown. But text/image reference is a good way to show user intent, but it isn't fully exploited in the method. Probably the intro can be updated to better show the motivation of the work.

* Typo in table 2 intend ->intent

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes a novel interactive segmentation framework that utilizes the model's intrinsic knowledge to segment ambiguous objects effectively.

### Strengths
1 The question is interesting, and the idea is well organized.

### Weaknesses
1. More visualized comparison results with state-of-the-art models should be added.
2. Please also add a time comparison with state-of-the-art models 
3. Please revise the writing of the methods part to make it easy to follow

### Questions
See the weakness

### Soundness
3

### Presentation
3

### Contribution
3

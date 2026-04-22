# Towards Pixel-level VLM Perception via Simple Points Prediction

- Avg Score: 4.00
- Decision: Reject
- Scores: 2, 2, 4, 6, 6

## Abstract
We present \textbf{\name}, a strikingly simple yet highly effective approach to endow Multimodal Large Language Models (MLLMs) with native pixel-level perception. 
Our method reframes segmentation as a simple sequence generation problem: the model directly predicts a \textbf{sequence of points} (textual coordinates) delineating object boundaries, entirely within its language space. 
To achieve high fidelity, we introduce a two-stage SFT→RL training pipeline, where Reinforcement Learning with an IoU-based reward refines the point sequences to accurately match ground-truth contours. 
We find that \emph{\textbf{the standard MLLM architecture possesses a strong, inherent capacity for low-level perception}} that can be unlocked without any specialized architecture. 
On segmentation benchmarks, \name achieves performance that is comparable to, and often surpasses, methods relying on complex, task-specific designs. 
This work lays out that precise spatial understanding can emerge from simple point prediction, challenging the prevailing need for auxiliary components and paving the way for more unified and capable VLMs. Code, data and
model are publicly accessible at \url{https://github.com/simpleseganonymous/SimpleSeg}.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes SimpleSeg, which models segmentation as predicting the vertex coordinates of object boundaries, thereby avoiding the use of complex segmentation decoders. The paper also trains the model using reinforcement learning (RL) with an IoU-based reward to improve performance.

### Strengths
1. Exploring how to perform segmentation using the MLLM's inherent capabilities, rather than relying on a segmentation decoder, is a meaningful research direction.
2. Predicting text-based coordinates allows for a unified training objective with the MLLM's other tasks.

### Weaknesses
1. Using polygons to model segmentation introduces approximation errors and limits expressive power. For example, in Figure 2, a polygon with a finite number of vertices cannot perfectly fit the line graph and the lightning bolt. Furthermore, the paper only considers cases with a single object boundary, whereas in real-world scenarios, many objects and backgrounds (e.g., a donut) cannot be represented by a single polygon.
2. Predicting text-based coordinates requires outputting a large number of text tokens, which significantly increases inference time.
3. The paper lacks comparisons with some recent works, including VistaLLM[1], HiMTok[2], UFO[3], and SAM4MLLM[4]. Notably, neither VistaLLM nor UFO requires a segmentation decoder, and VistaLLM also predicts polygon vertices.
4. The paper's segmentation performance shows a significant gap compared to state-of-the-art decoder-based methods.

[1] Jack of all tasks master of many: Designing general-purpose coarse-to-fine vision-language model. CVPR 2024

[2] HiMTok: Learning Hierarchical Mask Tokens for Image Segmentation with Large Multimodal Model. ICCV 2025

[3] UFO: A Unified Approach to Fine-grained VisualPerception via Open-ended Language Interface. NeurIPS 2025

[4] SAM4MLLM: Enhance Multi-Modal Large Language Model for Referring Expression Segmentation. ECCV 2024

### Questions
1. The method is only evaluated on Kimi-VL. Can it be extended to QwenVL or InternVL?
2. After the SFT and RL training described in the paper, what are the changes to the model's general VQA performance?
3. Does the model support multi-object segmentation?

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper enhances the capability of MLLM in the pixel-level prediction task via task reformulation and reinforcement learning. It modules the localization as a 4-tuple [text, point, box, mask], where mask is represented by polygon with a consistennt clockwise ordering. The RL algorithm enhances the learning flexibility of this mask representation.

### Strengths
It is novel that the paper incorporates reinforcement learning into pixel-level prediction of MLLM.

### Weaknesses
1. The mask representation of a polygon with the clock-wise vertex order is not new, which has been done in PolyFormer [ref1].

2. The data construction is not clear enough. Lack of implementation details, which reduces the reproducibility. What is the total number of training images? What is the data composition of SFT and RL training? What is the source data? Simply calling web data is not acceptable. What are the model versions of Grounding-DINO, SAM, and VLM? And there is no open-source statement for the data construction pipeline.

3. What is the sequence of the three training stages? What is the training/inference time cost and the GPU model. What is the exact meaning of Pre-train in Table 3?

4. Page 6 line 312, what is the meaning of ''thin-structure adherence''? Page 8 line 418, it cannot be seen from Fig. 4 that the performance is 53.7 cIoU at 78 tokens. It is below 40.

Considering the above questions and missing parts, I hold a negative rating towards this paper. I might change my mind if these questions are addressed well.

[ref1] Liu J, Ding H, Cai Z, et al. Polyformer: Referring image segmentation as sequential polygon generation. CVPR 2023: 18653-18663.

### Questions
see weaknesses

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors present a two-stage training process that manages to enable MLLMs to achieve significantly improved capabilities on pixel-level perception tasks. Via a combination of SFT and RL, the authors fine-tune models by casting pixel-level problems like segmentation into a sequence-of-point-coordinates format, which effectively allows solving such problems in the language space native to MLLMs, without the need for task-specialised decoder modules.

### Strengths
**Originality & Significance:**  
- The authors do a great job in outlining why the task matters, as well as how it is approached; all in all a very clear motivation of the presented research
- Simple yet elegant approach that requires no architectural modification of MLLMs to solve (referring expression) segmentation and comprehension
- Good results on the benchmarks, even in the context of architecturally-tailored approaches

 **Quality:**  
- The work is placed well within related efforts, and the specific gap the authors tackle is clearly presented and justified
- Appropriate extent of ablation studies to illustrate value/contribution of core components

**Clarity:**
- The work is mostly well written and easy to follow, and a good mix of illustrations and text makes the core concepts and expected results easy to grasp

### Weaknesses
- **Justification for use of RL is lacking** and incomplete (imo even misleading), see questions.

- **Several claims are made in a very broad manner** (mainly beyond the focus of this work), and would benefit from being toned-down a bit as they feel overstated (and are not necessarily being substantiated in the paper): 
  - E.g. l. 170 *'…can be seamlessly […] integrated as a new, core pre-training task for foundation models'* -> Note that this is quite a big claim, as the authors train/fine-tune a foundation model to solve ONLY this specific task; To substantiate this claim of being a valuable and ‘seamless’ part of pre-training, it would require demonstrating that using this task during pre-training doesn’t result in conflict and/or degradation of other qualities of the model. 
  - *'We pioneer the use of RL to optimize the entire generated sequence of points'* (RL is common to refine a sequence of tokens/points); I understand it’s meant in the context of segmentation, but again feels a bit ‘overstated’ (given that there are other works that use RL for segmentation, albeit with different setups & backbones)
  - *'in essence, reinforcement learning is a more reasonable and efficient. Optimization method for perception tasks'* -> Again, this is a very big and broad statement that is not substantiated at this scale; Stating RL as the ‘most efficient’ way of optimising is also a rather surprising statement given the sparse supervisory signal and hence data inefficiency it provides.

  $\Rightarrow$ While I do see what the authors are getting at, such broad statements should only be made if they are substantiated in the paper, and would otherwise benefit from being toned-down a little / from being made more concise and specific


- **Quality/Attention to detail should be improved**, e.g. 
  - l. 419/420: ‘[…] moderate density (221 tokens) yields the best score’ -> Does not match Figure 4, where best score is reached at somewhere around 300 tokens 
  - incorrect prompts displayed in visualisations: e.g. prompt in Figure 2 lower-right, Fig 9 (‘dragon’ vs. ‘street’), 
  - repeated sentences in the text: e.g. l. 157
  - typo in Takeaway2 (‘generalization of on’) 
  - Figure 5 should be Table 4

### Questions
**TL;DR:** I see the merit of this paper, but have a number of questions/points I’d like the authors to address. I’m happy to reconsider my current rating based on the authors’ response to my questions:

- Justification for RL: “Contours are inherently many-to-one w.r.t. masks; enforcing exact token matching is suboptimal. Reinforcement learning well bridges the gap …”   
  $\rightarrow$ I’d like the authors to further elaborate on this aspect, as I think it is currently stated in a very misleading way. To me, the many-to-one nature isn’t actually a problem that’s bridged or solved via RL.   
  $\rightarrow$ Rather, the many-to-one mapping is circumvented by using an abstracting metric like IoU, which essentially judges the resulting overlap of the area rather than the point-sequence itself; which is, however, independent from RL.  
  $\rightarrow$ The MSE distance between the centroids used as second metric in the RL reward could directly be used as a supervised loss, again circumventing any direct point-sequence judgement;  
  $\rightarrow$ In my opinion, the main benefit of RL is that it solves the challenge of backpropagating gradients to the points when using the IoU, which isn’t trivial when trying to use ‘normal’ supervision given that the output is in the language space and therefore discretised/sampled; so RL bridges this disconnect, but not any many-to-one assignment problem; correct?  

- L. 255: What exactly is meant by “instantiate queries as Cartesian product of the available elements”? 

- Are the examples in Figure 2 results from the training or validation/test set?

- Multiple questions re: Table 3: 
  - What do the improvements (in parentheses) each actually refer to, i.e. what’s the baseline for each of these improvements?
  - L406ff: Benefit through pre-training; I can see that SFT benefits through pre-training by the stated 4.6 points (going from 65.5 to 70.1), but SFT+RL goes from 75.2 to 78.5 – which is far from the stated ’13.0’ improvement (rather 3.3); Or am I misinterpreting these?
  - Suggestion (optional): I’d recommend swapping the colours around, so that improvements are highlighted in ‘green’ and decrease is represented in ‘red’ (aligns more with the common association to ‘good’ and ‘bad’)

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The goal of the paper is to investigate whether a Multimodal Large Language Model (MLLM) can achieve a high-fidelity image segmentation by predicting mask contours as a sequence of points.

Authors formulate the promptable image (object instance) segmentation task as a sequence generation problem, where the input is a text + image prompt and the output is a sequence of points representing the corresponding (object instance) mask contour and show how to apply the SFT-RL training pipeline to train Kimi-VL to solve this problem.

The contributions of this work are as follows:
1) new MLLM training suitable for the promptable image (instance) segmentation, which "naturally unifies points, boxes, and masks under one textual interface";
2) decoder-free, architecture-agnostic SFT-RL image (object instance) segmentation training pipeline, which, as the authors say, is simple, yet shows strong results on referring expression benchmarks; 
3) ablations of the training pipeline-related hyperparameters.

### Strengths
1) Strong results on segmentation benchmarks.
2) Significance of task (wide range of practical applications)
3) Simplicity (easy to read and follow paper and implementation details, except points mentioned in Weaknesses (Presentation)).

Although I believe the central claim of the paper is not sufficiently supported by the evidence, given that the results can be applied to various practical applications (such as controllable image editing, vision-based tool use, and GUI-grounded agents), I consider the overall contribution of the paper to be valuable to the community.

### Weaknesses
1. The central claim (L92-93) "we demonstrate that standard MLLM architectures possess a strong, inherent capacity for fine-grained perception" is not sufficiently supported by evidence, as the approach is validated only on one VLM architecture -- Kimi-VL. This claim might hold for other architectures, but it was not tested in this work.

2. Some components are not presented clearly or explanation is missing (Presentation):
- L268-266: Not clear what "large scale web data" means in this context, how it was collected, and whether it complies with CoE.
- In L266-267 (Sec. 3.2 Training Pipeline), it is stated that training encompasses SFT and RL phases. However, in L400-401 (Sec. 4.3) and Table 3, pre-training without SFT is also mentioned. Not clear what is done in this context.
- In L319: "Training uses 32 NVIDIA GPUs" neither GPU type nor memory is specified.
- Supplementary materials visualizations lack explanations/interpretation. For example, in Figure 9 it is not clear whether text query "Please recognize and segment the dragon in the image." is duplicated twice for two different images intentionally. If so, what authors aimed to demonstrate by this.

3. Ablation studies are focused on hyperparameters (which is useful indeed) but lack experiments on segmentation results (for example, failure case analysis). See questions for more details.

4. Authors state that "Code, data and model are publicly accessible at https://github.com/simpleseganonymous/SimpleSeg." As of Monday, 27 October, the repository contains just two data preprocessing scripts and the README, which states: "We provide the data processing scripts for our SimpleSeg in data.py, which can generate json files in LLaVA conversation formats."

### Questions
1. Was a failure case analysis performed (on what objects approach does not perform well)? For example, how does it work if objects have wholes or when prompted to segment objects which does not appear in the image? It would also be interesting to see the distribution of errors between semantic (i.e., not capturing the correct object from the textual description) and geometric (i.e., inaccurate or invalid mask) errors.

2. How pre-training phase mentioned in L400-401 (Sec. 4.3) and Table 3 is done? 

3. Why does text grammar differ for different target types L263-267? Points, rectangles and polygons can be represented using [[x, y], ...] grammar.

4. L156-157: sentence repetition. L276: including -> includes?

### Soundness
2

### Presentation
2

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
This paper introduces SimpleSeg, a minimalist framework that enables MLLMs to perform pixel-level segmentation without any specialized decoders. The key idea is to reframe segmentation as a text generation problem, where the model predicts a sequence of 2D points representing an object’s boundary directly in language space. The training pipeline consists of two stages: SFT for basic grounding and RL with an IoU-based reward to refine contour accuracy. The authors demonstrate that standard MLLMs already possess latent low-level perceptual capacity, which can be unlocked through this pipeline.

### Strengths
1. Reframing segmentation as point-sequence generation within language space is a clear, novel, and elegant idea that aligns with the unified philosophy of MLLMs.
2. Despite its simplicity, SimpleSeg performs comparably or better than more complex models, demonstrating that architectural minimalism can achieve strong pixel-level perception.

### Weaknesses
1. While the concept is elegant, the technical novelty may be seen as moderate; the method largely combines known ideas (polygon prediction + RL optimization).
2. There are also other decoder-free projects, such as GiT[1] and UFO[2].  It would be better to include some discussion.
3. The evaluation focuses mainly on referring expression segmentation benchmarks; more diverse real-world datasets (e.g., COCO panoptic or open-domain segmentation) would strengthen generalization claims.

[1] "Git: Towards generalist vision transformer through universal language interface." EECV, 2024.

[2] "Ufo: A unified approach to fine-grained visual perception via open-ended language interface." NeurIPS, 2025.

### Questions
Can the method in the paper be generalized to the segmentation of multiple objects?

### Soundness
3

### Presentation
3

### Contribution
3

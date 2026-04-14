# Continuous Space-Time Video Super-Resolution via Event Camera

- Decision: Reject
- Scores: 5, 5, 3, 6

## Abstract
Continuous space-time video super-resolution (C-STVSR) aims to simultaneously enhance video resolution and frame rate at an arbitrary scale. Recently, implicit neural representation (INR) has been applied to video restoration, representing videos as implicit fields that can be decoded at an arbitrary scale. However, the highly ill-posed nature of C-STVSR limits the effectiveness of current INR-based methods: they assume linear motion between frames and use interpolation or feature warping to generate features at arbitrary spatiotemporal positions with \ubtxt{two} consecutive frames. This restrains C-STVSR from capturing rapid and \ubtxt{nonlinear motion} and \ubtxt{long-term dependencies} (\textit{involving more than two frames}) in complex dynamic scenes. In this paper, we propose a novel C-STVSR framework, called \textbf{HR-INR}, which captures both \textbf{h}olistic dependencies and \textbf{r}egional motions based on INR. It is assisted by an event camera -- a novel sensor renowned for its high temporal resolution and low latency. To fully utilize the rich temporal information from events, we design a feature extraction consisting of (1) a regional event feature extractor -- taking events as inputs via the proposed event temporal pyramid representation to capture the regional nonlinear motion and (2) a holistic event-frame feature extractor for long-term dependence and continuity motion. We then propose a novel INR-based decoder with spatiotemporal embeddings to capture long-term dependencies with a larger temporal perception field. We validate the effectiveness and generalization of our method on four datasets (both simulated and real data), showing the superiority of our method.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an event-guided method for simultaneously enhancing video resolution and frame rate at an arbitrary scale. It captures both holistic dependencies and regional motions based on INR. Experimental results show the superiority of the proposed method.

### Strengths
* The idea of simultaneously VSR and VFI using events is new. And the proposed method achieve the state-of-the-art (SOTA) performance on not only VSR but also VFI.
* The idea about using implicit neural representation (INR) could be new.

### Weaknesses
* Tech

  * The authors say they use INR for C-STVSR. However, as far as I know, INR is something like NeRF (\ie, using a neural network to represent something such as 3D/2D objects). However, in this paper the author do not define the INR in a clear manner. I cannot find where the INR is used. It seems that the term INR in this paper is about some feature layers used in the designed network architecture, not representations. It is much more like "designing a new network structure/proposing new modules" instead of propose a new INR. There are lots of works about event representation (for example,  PhasedLSTM [a],  MatrixLSTM [b] , NEST [c]). However, the authors do not mention, discuss, or compare with them.
  * How to define the regional and holistic events? It seems that they are just two handcrafted terms without clear meaning. The motivation of defining such terms is not clear.
  * It seems that the other event-based VFI methods can remove the motion blur. However, from Figure 1 we can see that the motion blur still exists.

* Experiments

  * The comparisons may be not that comprehensive enough. For example, the authors should compare with pipelines such as "SOTA event-based VFI method + SOTA event-based VSR method" (\eg,   CBMNet + EG-VSR and other combinations) in Table 1. Only in this way it can show that it achieves the SOTA performance on C-STVSR. It seems that the compared baselines do not involve such kinds of pipelines.
  * Why not evaluate on the GoPro and Adobe dataset for either VSR or VFI? It seems that the authors only evaluate the performance of C-STVSR on this two datasets. Besides, for VSR, why not evaluate the proposed method on ALPIX-VSR dataset in [d]?

* Writing

  * The writing quality is not that good. For example, the conference names in Ref are not unified (\eg, for CVPR, the authors use both "2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) " and " Proceedings of the IEEE/CVF conference on computer vision and pattern recognition").

  [a] Phased LSTM: Accelerating recurrent network training for long or event-based sequences 

  [b] M.: A differentiable recurrent surface for asynchronous event-based data

  [c] NEST: Neural Event Stack for Event-based Image Enhancement 

  [d] Learning Spatial-Temporal Implicit Neural Representations for Event-Guided Video Super-Resolution

### Questions
* What is the upper bound of the number of FPS? This should be compared with the other VFI methods. Besides, the performance of setting different numbers of FPS should also be analyzed.
* In the abstract, the authors mention that continuous space-time video super-resolution (C-STVSR) aims to simultaneously enhance video resolution and frame rate at an arbitrary scale. Therefore, can the video be SR in any scale using the proposed method? Please show an example with a large scale (\eg, 8x 16x).

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper presents a space-time video super-resolution method using an event camera. The proposed approach utilizes events and video inputs, employing an event temporal pyramid representation, a comprehensive event-frame feature extractor, and spatiotemporal decoding to produce high-frame-rate, high-resolution videos. Experimental results demonstrate that this method significantly outperforms existing state-of-the-art techniques, particularly in terms of temporal consistency, as shown in the supplementary videos.

### Strengths
1.	The proposed temporal pyramid representation captures motion information at multiple pyramid levels, allowing the subsequent feature extraction module to access a range of detail from sharp but sparse features to blurrier yet denser details.
2.	Two specialized feature extractors for events and frames are introduced, each designed to capture information at distinct temporal resolutions.
3.	The spatial-temporal decoding module, which integrates temporal and spatial embeddings, enhances the method’s temporal and spatial consistency, outperforming existing space-time video super-resolution techniques.

### Weaknesses
1.	The feature extractors appear to be based on a Swin-Transformer and U-Net (Swin-Unet) combination, which raises concerns about novelty; additional ablation studies would help validate their effectiveness.
2.	Given that this is a space-time video super-resolution approach, including one or two video metrics to assess quality and temporal consistency would provide a more comprehensive evaluation.
3.	The visual comparison results include only one or two baseline method, while more methods are compared in the tabular results.

### Questions
It appears that the proposed method can handle a longer sequence of events as input. In comparison, what is the length of the event input used by other methods in this study? Given that a long event sequence may not be suitable for other methods to provide optimal results, testing different sequence lengths would be beneficial.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper addresses Continuous Space-Time Video Super-Resolution (C-STVSR), which aims to upscale both resolution and frame rate in videos at arbitrary scales. Existing methods that use implicit neural representation (INR) struggle with capturing rapid, nonlinear motion, and long-term dependencies due to their reliance on linear motion assumptions and limited frame usage. To address these limitations, the authors propose HR-INR, a C-STVSR framework enhanced by event  signals, which provide high temporal resolution. The framework includes a regional event feature extractor that captures nonlinear motion through an event temporal pyramid and a holistic event-frame feature extractor for long-term dependencies. Additionally, an INR-based decoder with spatiotemporal embeddings is introduced to handle long-term dependencies across a large temporal scope. Results on both synthetic and real-world datasets demonstrate the proposed method’s effectiveness and generalization capabilities.

### Strengths
1. In the STVSR task, the paper effectively addresses the limitations of RGB-based methods in handling highly dynamic motion by introducing event signals, which helps improve performance.
2. Compared to existing event representation methods, such as voxel grids, time surfaces, and time moments, this paper proposes an innovative event representation method, the Temporal Pyramid Representation (TPR).

### Weaknesses
1. The paper claims that compared to existing RGB-based methods such as VideoINR and MoTIF, the use of event signals can address highly dynamic motion, which is intuitive. However, the assertion that extending the input window to four frames solves the long-term dependency issue lacks explanation and experimental validation. For instance, it would be helpful to clarify the specific scenarios where a four-frame window is essential, and a two-frame setup would be insufficient. A suggestion would be to visualize Temporal Attention on the input images in Figure 4.
2. While the introduction of event signals may address highly dynamic motion, the paper does not discuss any potential challenges introduced by event signals. The scientific problem being addressed also remains unclear. This paper seems to be engineering-oriented without scientific rigor. 
3. The paper claims its approach avoids the gaps and holes commonly associated with optical flow methods. However, in comparison with the optical flow-based VideoINR, no explanation or experimental evidence demonstrates how this approach mitigates these issues, or why VideoINR cannot mitigate them. For example, in Figure 13, the HR-INR method exhibits blurring in frame 000080 and noticeable holes in the hand region of frame 000160, indicating that gaps and holes persist in certain cases. As a result, the novelty of this work relative to previous studies is not established.
 
4. The contributions of the paper are not clearly defined. It is recommended that the authors refine and explicitly summarize the key contributions of the work in the writing.

Overall, I think 
1. the proposed method lacks a clearly defined contribution to the STVSR task. It seems to be too engineering-oriented. 
2. the experimental results are insufficient to demonstrate a clear advantage over RGB-based methods.

### Questions
1. Why does Figure 8 compare the proposed method with the sub-SOTA method, TimeLens, rather than the more recent SOTA method, such as CBMNet?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Summary: 

This paper addresses the challenging task of Continuous Space-Time Video Super-Resolution (C-STVSR), which aims to enhance video resolution and frame rate at arbitrary scales. Experiments demonstrate HR-INR’s effectiveness across four datasets, achieving state-of-the-art results.

### Strengths
Strengths

1.The authors’ novel approach of integrating an event camera for high-resolution temporal information is commendable. Event cameras capture finer temporal details than traditional cameras, making them particularly effective for dynamic scenes with nonlinear motion. This approach allows the method to move beyond traditional frame-based video super-resolution limitations.

2.The model is tested on a variety of datasets, both simulated and real, highlighting its robustness and generalization ability. The results indicate HR-INR’s superiority over prior methods, even achieving better performance with much fewer parameters than other models.

### Weaknesses
Weaknesses:

1.It appears that this paper achieves better performance and accomplishes more tasks with fewer parameters (e.g., one-tenth of TimeLens’s parameters). Can the authors analyze whether their method has surpassed existing methods in both super-resolution and interpolation tasks? Are there any limitations?

2.In the experimental setup, the authors used Adobe24 + Vid2e to generate events for pretraining and tested on other datasets. If I’m not mistaken, the training data for other methods, aside from the CED dataset, is different from that used in this paper. Although the authors emphasize that TimeLens used a larger training set, it seems that event density, which affects the performance of event-based tasks, was not standardized across methods.

3.For the analysis of the Temporal Representation Pyramid (TRP), is there actually motion within 1/1000s in the current datasets?

### Questions
Questions:

Considering that this paper does not commit to open-source code, do the authors have any plans to release the code?

### Soundness
3

### Presentation
3

### Contribution
2

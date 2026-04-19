# Towards Ultra-High-Definition Image Deraining: A Benchmark and An Efficient Method

- Decision: Reject
- Scores: 5, 6, 8, 5

## Abstract
Despite significant progress has been made in image deraining, existing approaches are mostly carried out on low-resolution images. The effectiveness of these methods on high-resolution images is still unknown, especially for ultra-high-definition (UHD) images, given the continuous advancement of imaging devices. In this paper, we focus on the task of UHD image deraining, and contribute the first large-scale UHD image deraining dataset, 4K-Rain13k, that contains 13,000 image pairs at 4K resolution. Based on this dataset, we conduct a benchmark study on existing methods for processing UHD images. Furthermore, we develop an effective and efficient architecture (called UDR-Mixer) to better solve this task. Specifically, our method contains two building components: a spatial feature rearrangement layer that captures long-range information of UHD images, and a frequency feature modulation layer that facilitates high-quality UHD image reconstruction. Extensive experimental results demonstrate that our method performs favorably against the state-of-the-art approaches while maintaining a lower model complexity. The code and dataset will be available to the public.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
This paper focuses on ultra-high-definition (UHD) image deraining and introduces the first UHD image deraining dataset, 4K-Rain13k. The authors also propose a dual-branch network architecture, UDR-Mixer, to better address this task. Specifically, a spatial feature rearrangement layer is employed to capture long-range information in UHD images, while a frequency feature modulation layer complements the reconstruction of UHD image details. Qualitative and quantitative experimental results demonstrate that the proposed method outperforms existing state-of-the-art approaches on both the proposed dataset and several public datasets.

### Strengths
1. The proposed dataset could benefit the community and inspire further research.
2. The experiments and ablation studies are comprehensive, demonstrating the effectiveness of the proposed method compared to existing state-of-the-art approaches.

### Weaknesses
1. The technical contribution of the proposed method is limited. The proposed Spatial Feature Rearrangement Layer (SFRL) is quite similar to the Global Feature Modulation Layer (GFML) in MixNet [1].  The author should explicitly compare and contrast the two modules. Adding a paragraph to discuss the key similarities and differences, and highlighting the novel design of the proposed SFRL, is essential.
2. Many implementation details are unclear and not described in the paper. For instance, the authors mention that raindrop generation is modeled as a motion blur process to synthesize corresponding raindrop images. However, how exactly are these images generated? Additionally, the authors claim that alpha blending is used to ensure fidelity, but how are the blending weights determined, and how do different weights affect fidelity?  The authors should include a more detailed subsection on the dataset generation process, covering the exact motion blur modeling and alpha blending techniques used. To provide a clearer understanding, examples showing how different blending weights affect the fidelity of the final synthesized images should also be provided.
3. To better demonstrate the efficiency of the proposed method, the comparison of runtime should be reported. The author should include a table or figure comparing the runtime of the proposed approach to other SOTA methods on a specific hardware setup.

[1] Wu et al., MixNet: Efficient Global Modeling for Ultra-High-Definition Image Restoration. arXiv2024.

### Questions
The authors should address the issues mentioned in the weaknesses, particularly those related to the technical design and implementation details.

### Soundness
2

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
3

### Summary
The paper introduces a 4K dataset with synthetically generated rain streaks, for the purpose of data-driven de-raining image enhancement. The dataset contains 13K images and is produced by taking some of the specific challenges in synthesizing rain at this resolution level into account. An MLP-Mixer-based network is formulated, with the aim to allow for de-raining of UHD images with less computational demands compared to previous state-of-the-art methods. The architecture is designed using separate branches for spatial and frequency domain processing. Low-level image features are processed in an autoencoder network composed of spatial feature mixing blocks, while an auxiliary branch performs frequency domain mixing to promote quality at the UHD resolution. Experiments compare against previous methods that are applicable in the UHD domain, on the synthetic images with reference-based metrics and on real rainy images using non-reference quality metrics. Additional experiments also compare on datasets of lower resolution, where the proposed method is adapted to work better at this resolution. An ablation study explore the different components of the network, e.g., different strategies for mixing.

### Strengths
The focus on UHD image de-raining is a natural extension of previous work, and there is good value in providing a dataset for this purpose. The proposed design focusing on efficient processing at the UHD resolution level is demonstrated to be both efficient and generating competitive performance in comparison to previous methods. Although this is an adaptation of previous work, there seem to be some novel aspects in terms of promoting detail reconstruction at high resolution. Overall, the combination of UHD focus, providing a dataset for this purpose, and competitive results, makes for relatively high significance.

### Weaknesses
One of the formulated contributions is the dataset. However, in its current form the paper falls short of demonstrating this as a strong contribution. There is very little detail regarding the dataset. Online sources are not specified, and the simulation of rain is under-explained. The motivation for why the simulation is different in UHD images is unclear and speculative, and there is very little information on the transformations used to account for this in the simulation. It would not be possible to reproduce the simulation without a significant amount of additional details. 

While the paper has an ok structure overall, the writing could be improved in terms of formulations and grammar. Also, the reference format needs to be revised (differentiating between \citet and \citep depending on usage).

### Questions
* The comparison to MAXIM is only performed in terms of one example image. How does the proposed model compare in terms of PSNR on the Rain13k dataset?
* How was the selection of previous methods done? Specifically, how did you determine that a model is not possible to use on the UHD dataset? Is it due to architectural constraints, memory consumption, or computational complexity? If memory or computations is a deciding factor, what is the threshold for deciding if it is not applicable to UHD?
* In relation to training of the compared methods, it is explained that "We uniformly select the weights from their final training epoch for testing purposes". Does this mean that all models, including the proposed, are trained for an equal number of epochs?
* How was other hyper-parameters tuned for the different models? Are all methods trained using their respective default settings? Wouldn't it potentially be other optimal hyper-parameters when training on the UHD dataset?
* Some limitations are mentioned in the end of the paper. It would be interesting to see some examples of typical failure cases. Are there some other notable artifacts in specific situations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper contributes the first large-scale UHD image deraining dataset and proposes MLP-based architecture (UDR-Mixer) to achieve this task. Extensive experiments demonstrate that UDR-Mixer performs favorably against the state-of-the-art approaches.

### Strengths
1. The paper constructs the first high-quality UHD image deraining dataset (4K-Rain13k).
2. The paper develops a dual-branch architecture UDR-Mixer, where the spatial feature mixing block and the frequency feature mixing block are proposed.
3. Experimental results demonstrate that UDR-Mixer achieves a favorable trade-off between performance and model complexity.

### Weaknesses
1. It is more appropriate to calculate #FLOPs on 4K images.
2. The testing time on 4K images can also be given.
3. For Lines 362-363, why not evaluate by cropping the image into multiple patches?
4. In Table 3, the authors can use some more advanced non-reference IQA metrics, e.g., MANIQA, MUSIQ, and CLIPIQA.
5. It is better to give some 4K deraining results of UDR-Mixer trained on low-resolution datasets.

### Questions
Please see 'Weaknesses'.

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
5

### Summary
This paper addresses the task of deraining UHD images, which is not yet well-explored despite advances in low-resolution image deraining research. The authors introduce the first large-scale UHD image deraining dataset, containing 13000 pairs of 4K resolution images. Using this dataset, they benchmark existing methods and identify performance limitations for UHD images. Additionally, the paper presents a model called UDR-Mixer for a balance of both effectiveness and efficiency. Experimental results show that the proposed method outperforms state-of-the-art methods with lower model complexity, making it suitable for practical use.

### Strengths
- The paper introduces the first large-scale UHD image deraining dataset, 4K-Rain13k, filling an the gap in high-resolution deraining research
- The method is evaluated on both synthetic and real dataset, which verifies its practical use.

### Weaknesses
- I do not really understand how oversized models are evaluated from the expressions in Line 361- 363. If I do not misunderstand, the authors first resize the image to the largest size that can be processed by a single GPU, and then perform these deraining models, finally enlarge the size to the original size and evaluate the metrics. If so, I concern it would be unfair for these compared methods. Changing the resolution or dpi during the inference would greatly reduce the performance, since the model is not trained on this resolution, and resize operation would also lead to blurry artifacts. A more rigorous way is to divide the input image to several overlapped patches, evaluate on each patch, and then combine them.
- I am not sure whether it is necessary to have a training dataset with 4K resolution. We cannot send the full resolution images during training, just like this paper crop $768 \times 768$ patches for training. So from my view, it is equivalant to have a training dataset with $768 \times 768$  resolution. Please explain the rationale behind using 4K resolution for the training dataset given the $768 \times 768$ patch size used in training.
- I concern whether the rain streak generation approach is authentic enough to generate 4K images. With higher resolution, the details of real rain streak (like the texture, position, perspective relationship) should be clearer than that in a low-resolution image. It requires more realistic generation method for high resolution. However, in Fig. 2, the generated rain streak seems like a separate layer in front of the scene, and the discrepencies are enlarged in high resolution images.
- The dataset is regarded as an important contribution to this paper. Therefore, the authors should also conduct ablation studies on dataset, to verify why 4K training dataset instead of low-resolution training dataset is important for 4K deraining testing under synthetic/real scenarios. Specifically, conduct ablation studies comparing models trained on 4K data vs. lower resolution data, but tested on 4K images.

### Questions
Please see my comments in the weakness and answer the questions I raised in the weakness.

### Soundness
3

### Presentation
3

### Contribution
2

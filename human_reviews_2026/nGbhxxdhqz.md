# Maximizing Asynchronicity in Event-based Neural Networks

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 6

## Abstract
Event cameras deliver visual data with high temporal resolution, low latency, and minimal redundancy, yet their asynchronous, sparse sequential nature challenges standard tensor-based machine learning (ML). While the recent  asynchronous-to-synchronous (A2S) paradigm aims to bridge this gap by asynchronously encoding events into learned features for ML pipelines, existing A2S approaches often sacrifice expressivity and generalizability compared to dense, synchronous methods. This paper introduces EVA (EVent Asynchronous feature learning), a novel A2S framework to generate highly expressive and generalizable event-by-event features. Inspired by the analogy between events and language, EVA uniquely adapts advances from language modeling in linear attention and self-supervised learning for its construction. In demonstration, EVA outperforms prior A2S methods on recognition tasks (DVS128-Gesture and N-Cars), and represents the first A2S framework to successfully master demanding detection tasks, achieving a 0.477 mAP on the Gen1 dataset. These results underscore EVA's potential for advancing real-time event-based vision applications.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the sparsity and asynchrony of event camera data by proposing an Asynchronous-to-Synchronous (A2S) framework to better align event streams with deep learning methods. Inspired by language models, the authors treat event streams as ordered and incrementally evolving information, introducing Event Asynchronous Feature Learning (EVA), which leverages linear attention and self-supervised learning. Experimental results demonstrate that the proposed method achieves strong performance on both event-based recognition and inspection tasks.

### Strengths
1. The motivation is clear. This paper specifically addresses the sparsity and asynchrony of event camera data and proposes a targeted solution. Treating continuous events as ordered natural language sequences and tokenizing them is a novel idea.

2. The proposed method plays a positive role in bridging event camera data with deep learning approaches, effectively promoting progress in this research area.

3. The writing is fluent overall, with a clear and coherent logical structure.

### Weaknesses
1. Insufficient experimental datasets. For the object recognition task, the authors only used relatively simple datasets such as N-Cars and DVS128-Gesture, without evaluating the method on more complex datasets like N-Caltech101 and N-ImageNet. This limitation may reduce the persuasiveness of the proposed approach.

2. Low resolution of experimental datasets. The datasets used—N-Cars, DVS128-Gesture, and Gen1—all have relatively low resolutions, whereas modern event cameras can already reach 1280×720. Therefore, it is highly necessary to test the proposed method on high-resolution datasets. Moreover, during the A2S conversion, the resolution of the event camera can significantly influence the transformation speed. The authors should discuss how resolution affects the performance and efficiency of their method.

### Questions
1. From Figure 4, it appears that MRP and NRP are not directly used in downstream tasks, but rather serve to guide the training of the event embedding module?

2. In Figure 3C, the event embedding is transformed by the task head into a representation, which is then used to compute the loss together with features derived from MRP and NRP. How are MRP and NRP themselves trained in this setup?

3. During the self-supervised learning process, how are MRP and NRP trained using handcrafted features? Are they trained separately in advance? Does this pretraining need to be done for each task or dataset individually? Why not directly use handcrafted features to train the event embedding instead?

4. In Figure 6, what exactly are the learned representations, and what insights do they provide?

5. Is the representation generalizable to all event-based downstream tasks, including optical flow estimation, semantic segmentation, etc., or is it restricted to specific tasks only?

6. Compared with directly stacking event data into event images and processing them, what advantages does the proposed method offer? Does it provide benefits in terms of inference speed or computational cost (FLOPS)?

### Soundness
3

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
5

### Summary
This paper presents EVA, a novel asynchronous-to-synchronous (A2S) processing framework for event cameras. EVA builds an asynchronous encoder based on a linear attention mechanism and leverages self-supervised learning to learn generalizable event representations. It significantly outperforms existing A2S methods across multiple event-based vision tasks, including object recognition and object detection. Notably, EVA achieves 47.7 mAP on the Gen1 detection dataset, and demonstrating its strong representation capability and generalization across complex tasks.

### Strengths
1. The paper draws an insightful analogy between event streams and language sequences, introducing a linear attention mechanism and self-supervised learning to propose a novel and efficient paradigm for asynchronous event processing.

2. The authors conduct extensive evaluations on multiple public datasets (DVS128-Gesture, N-Cars, and Gen1), covering both recognition and detection tasks. The results are convincing and demonstrate clear advantages over various baseline methods.

3. The manuscript is well written and logically structured, with detailed figures and tables, and a comprehensive appendix that further supports the work.

### Weaknesses
1. As an A2S framework, a key question is how to achieve better asynchronous event representation. While I acknowledge that the authors have already included sufficient baseline comparisons, I wonder whether they have considered fixing the task-specific backbone and varying only the event representations, as described in Section 2.1. For example, in the object detection task on Gen1, one could fix a backbone (e.g., RVT-B) and compare different existing representation methods (e.g., time-surface, Matrix-LSTM, EST, voxel grid, graph-based). You may refer to Table 1 in ACGR [1] for a relevant comparison format.

2. Although Appendix H provides latency analysis, it focuses only on comparisons within EVA itself. It would be more informative to include approximate inference time and computational complexity comparisons across different open-source representation methods (e.g., Matrix-LSTM, EST, voxel grid, graph-based). Since event representation is a fundamental technique, demonstrating its low-latency characteristics is particularly important.

3. In future journal extensions, the authors could consider including more visualization results in the main text to better highlight the advantages of the proposed method. 

[1] Asynchronous Collaborative Graph Representation for Frames and Events, CVPR 2025.

### Questions
1. The fitting targets during self-supervised training in this paper are EC and TS. Ideally, after training is completed, the proposed method could become some kind of weighted mixture of EC and TS. So why not directly use EC and TS? For example, in the experiments presented in this paper, what would be the performance change if the input representation were replaced with EC and TS?

2. Integration of events and RGB frames is an emerging trend. Could this framework be extended in the future to design a joint asynchronous representation that fuses both modalities? It would be interesting to hear the authors’ thoughts or potential directions on this.

3. Introducing NLP-inspired approaches into event representation is a novel idea. What specific advantages does this bring compared to conventional designs? Could the authors provide a more theoretical explanation or justification for this perspective?

4. Looking ahead, could this asynchronous representation technique be incorporated into event-based multimodal large language models [2]? This would be an exciting direction for exploring the intersection between event-based vision and multimodal reasoning.

[2] EventGPT: Event Stream Understanding with Multimodal Large Language Models, CVPR 2025.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper introduces EVA (EVent Asynchronous feature learning), a novel Asynchronous-to-Synchronous (A2S) framework for learning expressive and generalizable representations from event streams. The key idea is to treat events as “language tokens” and borrow concepts from language modeling, including linear attention and self-supervised feature learning, to construct a unified A2S architecture. EVA demonstrates strong results on multiple benchmarks—DVS128-Gesture and N-Cars for recognition tasks—and becomes the first A2S framework capable of performing detection, achieving 47.7 mAP on the Gen1 dataset.

### Strengths
The proposed framework is well-structured and systematically designed. The overall pipeline is coherent, covering event tokenization, temporal-difference encoding, MVHS feature generation, and self-supervised tasks in an end-to-end manner.

The experimental evaluation is comprehensive; the paper validates the generality of EVA across several datasets and tasks, includes meaningful ablation studies, and provides detailed experimental configurations.

The conceptual alignment between event representation and language modeling is novel and promising, opening opportunities for future research on cross-modal modeling between event-based vision and sequence learning.

### Weaknesses
In the supplementary material (Fig. 6(c)), it is unclear whether the visualized features correspond to the MVHS outputs. Section E. Visualization lacks a clear description or interpretation of these results.  

While performance metrics on N-Cars and Gen1 are reported, latency, throughput, and scalability analyses are missing. Since A2S methods can, in principle, operate at arbitrary event sampling frequencies, it is important to analyze how performance and efficiency vary with sequence length.  

There are also minor presentation and clarity issues. The term “#patches” appears in Section 3.1.3 without prior definition and only reappears in the supplementary material. The notation \(P_r\) in Eqs. (4) and (5) is undefined, and Table 2 is never cited in the main text.

### Questions
How is the number of events \(N\) in MVHS determined for different datasets, and how does it affect performance and computational cost?  

The motivation for selecting Event Count (EC) and Time Surface (TS) as the self-supervised prediction targets should be clarified, as well as whether other representations have been considered.  

Although EVA avoids event accumulation, Table 1 shows an additional latency of 14.7 ms. Comparisons with other learnable event representations such as ERGO (“From Chaos Comes Order”) and EventPillars (“Pillar-based Efficient Representations for Event Data”) under identical backbones would better position EVA’s efficiency and expressivity.

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
3

### Summary
This paper introduces EVA, an asynchronous architecture for event camera data. The key idea is to treat streams of visual events as analogous to language tokens, each carrying incremental information about a scene and to employ linear attention architectures for scalability. The encoder maintains a hidden state that aggregates spatial and temporal structure while supporting fast processing.

### Strengths
* Tackles a timely challenge as it is yet unclear how to process event camera inputs
* Interesting ideas to use a token based approach with a linear transformer
* The self-supervised aspect with the multi-representation training is seemingly novel and interesting

### Weaknesses
Overall, the major concerns I have with this work are that the results are not very compelling.

The authors only evaluate two task (object detection and recognition). Given the authors are proposing a new architecture, it would be good to have at least another slightly different task to demonstrate the robustness of their architecture is not clear. 

The proposed approach achieves does not achieve the best accuracy or latency, but is another tradeoff point in the space. 

If the key benefits are the fast processing, then it would be good to have a more thorough analysis of where the efficiency benefits come from and whether real-time performance can actually be achieve with their approach on realistic GPUs for edge computing.

The effectiveness of the approach with self-supervised learning using handcrafted representations is not convincing as the method to have it learn the right representations.

### Questions
Why is your latency higher when more parameters are used in Table 1?

### Soundness
3

### Presentation
3

### Contribution
2

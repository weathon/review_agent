# MORPH: Shape-agnostic PDE Foundation Models

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
We introduce MORPH, a modality-agnostic, autoregressive foundation model for partial differential equations (PDEs). MORPH is built on a convolutional vision transformer backbone that seamlessly handles heterogeneous spatiotemporal datasets of varying data modality (1D--3D) at different resolutions, and multiple fields with mixed scalar and vector components. The architecture combines (i) component-wise convolution, which jointly processes scalar and vector channels to capture local interactions, (ii) inter-field cross-attention, which models and selectively propagates information between different physical fields, (iii) axial attentions, which factorize full spatiotemporal self-attention along individual spatial and temporal axes to reduce computational burden while retaining expressivity. We pretrain multiple model variants on a diverse collection of heterogeneous PDE datasets and evaluate transfer to a range of downstream prediction tasks. Using both full-model fine-tuning and parameter-efficient low-rank adapters (LoRA), MORPH outperforms models trained from scratch. Across extensive evaluations, MORPH matches or surpasses strong baselines and recent state-of-the-art models. Collectively, these capabilities present a flexible and powerful backbone for learning from the heterogeneous and multimodal nature of scientific observations, charting a path toward scalable and data-efficient scientific machine learning.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a new autoregressive foundation Model called MORPH, which is able to be pretrained on datasets with different numbers of spatial dimensions and fields. The model is based on a vision transformer and employs patching and axial dimension. The experiments show that the model, with the used pretraining datasets, can beat models trained from scratch and other foundation models.

### Strengths
1. Paper is generally well-written and easy to understand.
2. Showed the advantage of using LoRA for PDE foundation models.
3. Model architecture that can deal with different amounts of spatial dimensions as well as fields.

### Weaknesses
1. The main paper is too technical. It should focus more on the presented architecture and further experiments. The method section starts with a discussion of the data format. The introduced data format, seems to be however, not different from the standard format implicitly used in all PDE papers (besides maybe interpreting scalar fields and vector fields in the same format). Hence, it would be good to only mention the chosen input format shortly. Similarly, the discussion of the dataloader and whether it is better to apply normalization on the data directly or on the fly could be shortened or omitted in the main part.  Instead, it might also be worthwhile to add ablation and scaling experiments from the appendix to the main part.
2. While the paper is evaluated against other foundation models, the other foundation models are not pretrained with the same datasets. It would still be necessary to evaluate against other PDE foundation models when all have the same pretraining data. Otherwise, we cannot evaluate if MORPH generalizes better than the other models. Of course, Morph is highlighted as a model that can learn across a number of spatial dimensions. Hence, it would make sense to have an experiment where MORPH and  Poseidon, for example, are pretrained on the same 2D data, and another one where you add the 3D and 1D data to show how much MORPH can profit from the other data sources.
3. The second experiment mentioned in 2 would also show if MORPH can really profit from including other-dimensional data. This is also missing in the paper right now in general. It is only shown that MORPH is better when being pretrained vs. from scratch in general, but the pretrained set contains datasets with all spatial dimensions. Having an experiment showing that pre-training with 2D data helps on 3D fine-tuning sets would be very insightful. 
3. "Shape" as a word can be confusing since it could also be referring to airfoil shapes in the PDE world, for example, and not necessarily to the tensor shape.

### Questions
1. How is the patching done? Do you have a patch encoder for each number of spatial dimensions to project down to the same size across spatial dimensions?
2. What do you mean by partial observability in line 43? How does MORPH approach partial observability?  
3. What do you mean by attention "chaos" in line 99?

### Soundness
2

### Presentation
2

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
The paper introduces MORPH, a PDE foundation model which is targeted towards heterogeneous data, covering 1D, 2D and 3D at different resolutions. MORPH uses a unified physics tensor format for processing 1D-3D data with different fields. MORPH employs spatial + temporal axial attention as well as a inter-field cross-attention operation. Experiments include comparisons against standard benchmarks and recent foundation-like models such as DPOT, MPP and Poseidon in zero-shot and full-shot settings.

### Strengths
- Network architectures for PDEs that can be trained on a combination of 1D, 2D and 3D data with different fields and discretizations is a very timely and relevant problem
- In principle, the proposed network architecture combined with adequate positional embeddings makes a lot of sense and is a sound solution to this problem

### Weaknesses
- I am skeptical about the main experiment. For the "standalone surrogate" part, FNO and UNet are standard baselines. What specific U-Net architecture was used? U-Nets are often misrepresented in PDE benchmarks, because more modern variants are not used (e.g. any recent U-Net architecture in a high-impact paper on generative modeling for images). Other sota architectures (MPP, DPOT) are not trained on all datasets for the standalone surrogate (did you use the metrics from previous papers or retrained the models). The metrics seem very noisy; sometimes the smaller models (Ti, S) perform better, sometimes the larger networks (M, L). I would expect the scaling law to hold (larger models perform better), which was also shown in other PDE foundation-like model papers. Did you use the EMA weights for evaluation? 

- Continuing on the main experiment, but the "foundation models" part; DPOT and Poseidon were each pretrained on different pretraining datasets (the ones from the original paper) than MORPH? In this case, that would make the comparison difficult, since the pretraining datasets would be another crucial factor for the performance. Also, I am not sure if I would agree that when MORPH is trained on all pretraining datasets, then evaluating them on each of them individually can be considered a "zero-shot" setting. If all foundation models were pretrained on the same datasets, that would make the comparison much more useful, because then it would be possible to assess more clearly how well the transfer from pretraining datasets to finetuning datasets works. 

- The main paper contains a lot of technical material that can be moved to the appendix in my opinion to make space for more relevant ablation studies and experiments. For example, details about the data stream and sharding setup as well as  the compute infrasctructure. 

- While 3D datasets are considered, the resolutions are not very high. I expect the MORPH to scale well to high resolutions due to the axial attention, however I would appreciate a detailed model comparison regarding GFLOPs, throughput and VRAM requirements of the different networks. 

- The architecture of MORPH seems to be very similar to MPP (axial attention over space+time). Can you clarify the differences and if they are very similar, do an ablation study for every modification? I would also appreciate an ablation study on different design choices of the network. 

- The results on autoregressive rollouts are not comprehensive; are the models stable for more than 10 autoregressive steps? The dynamics in Figures 4 to 7 do not show much change over time. Why didn't you show rollouts for 3D? Those are most interesting as they are most challenging.  

- It's not true that "this is the first study to apply LoRA to PDE foundation models", see [1]

- The word "shape-agnostic" made me first believe the paper proposes a model that works on arbitrary geometries; the current title is overselling it to me as the model is still limited to regular grids. 

[1] https://openreview.net/pdf?id=3BaJMRaPSx

### Questions
- How is the input history of 10 previous states in DPOT-FM handled in the zero-shot setting?
- Have you considered any methods like Patch n' Pack https://arxiv.org/pdf/2307.06304 or others for training on different datasets and resolutions? 
- Have you looked at the performance of MORPH when changing the resolution of a specific pretraining dataset?
- I did not understand if you are using ST-Bilinear or S-Linear & T-Slice for MORPH? Have you considered relative positional encodings, e.g. RoPE? Do you think this makes sense? 
- The information density of a patch with patch size 8 is much lower for 1D than for 3D data. Is this a problem? 
- Do you use any additional conditioning information like a class label for each dataset? Would this be helpful?

### Soundness
3

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
3

### Summary
The paper introduces MORPH - a PDE foundation model that handles data of varying dimension at different resolutions and with mixed scaler and vector components.
The paper presents a flexible foundation model that can learn from diverse forms of partially observed scientific data.
The model supports shape agnostic with physics-aware channel handling of PDE datasets. This is done by using a) component-wise convolutions for local interactions b) interfield cross attention for modeling information across physical fields
c) axial attention for factorizing full spatiotemporal self attention along spatial and temporal axes, reducing compute while retaining expressivity. The paper defines a tensor format for minibatches that generalizes across PDE datasets while preserving semantics.
pretraining is done on six heterogeneous datasets. full finetuning and parameter efficient fine tuning done with LoRA on seven additional datasets. The largest model has 480M parameters.

### Strengths
PDE foundation models are an important development and this method attempts to overcome limitations of prior methods giving equal weight to heterogeneous data sources.

The method proposes a unified tensor format to handle sources of varying shapes and scalar and vector fields.

Model features are chosen for efficient and expressive processing of PDE fields where convolutional operations capture locality, fieldwise attention promotes interaction between fields and Factorized spatiotemporal attention in axial attentions makes for efficient processing.

Multiple models are trained with varying parameters counts and compared including with full and low rank fine tuning.

Experiments are performed on a number of datasets.

The paper is mostly clear and well written.

### Weaknesses
The unified tensor format handles fields that have varying the number of components by broadcasting to match vector components which appears to be wasteful.

The details of the operations of interfield attention and axial attention are vague from their description in the main paper. Some clarity would be useful.

### Questions
The paper mentions partial observability in the challenges with PDE modeling. How does MORPH handle this?

What mixed scalar and vector fields are used in the experiments? Do you assume at most 3d vector fields? What are the dimensions of the vector components?

line 217 how are the different data processing components combined into the transformer architecture. Is the flow sequential as convolution-> field wise -> axial -> decoding or is it more complex?

line 99. What is meant by attention chaos?

### Soundness
3

### Presentation
3

### Contribution
3

# SpikingVTG: Saliency Feedback Gating Enabled Spiking Video Temporal Grounding

- Decision: Reject
- Scores: 6, 6, 6, 5, 6

## Abstract
Video Temporal Grounding (VTG) seeks to retrieve consecutive intervals or specific clips from a video based on specified natural language queries. VTG requires accurately aligning video segments with corresponding natural language instructions, highlighting the need for effective methodologies to capture semantic correspondence and maintain temporal coherence. Spiking neural networks (SNNs), previously underexplored in this domain, present a unique opportunity to tackle VTG challenges from both the architectural and energy-efficiency perspectives. In this paper, we leverage sparse spike-based communication of SNNs to propose a multimodal architecture tailored for VTG tasks, namely SpikingVTG, providing a biologically inspired and efficient solution. Leveraging temporal saliency feedback, our proposed spiking video-language model (VLM) achieves competitive performance with non-spiking VLMs across diverse moment retrieval and highlight detection tasks. We introduce a Saliency Feedback Gating (SFG) mechanism that improves performance while reducing overall neural activity. To efficiently train our spiking VLM, we analyze the convergence dynamics of each neuronal layer and utilize equilibrium states to enable training using implicit differentiation at equilibrium. This approach eliminates the need for computationally expensive backpropagation through time while also enabling the use of knowledge distillation for efficient model training. To further improve operational efficiency and facilitate the on-chip deployability of our model, we leverage a multi-stage training pipeline that focuses on eliminating non-local computations, such as softmax and layer normalization, leading to the development of the Normalization Free (NF)-SpikingVTG model. Additionally, we create an extremely quantized variant, a 1-bit NF-SpikingVTG model, which vastly improves computational efficiency during inference while maintaining minimal performance degradation from our base model. Our work introduces the first spiking model to demonstrate competitive performance on VTG benchmarks, including QVHighlights and Charades-STA.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SPIKINGVTG, a SNN based Multimodel Transformer model for Video Temporal Grounding tasks. The author proposes a series of works to improve energy efficiency while reducing performance loss. 

Specifically, the core contribution includes:

1.Architecturally: SALIENCY FEEDBACK GATING (SFG): using intermediate temporal outputs to dynamically update the input to the model at every time step for better predictions.

2.Theoretical analysis: TRAINING LEVERAGING CONVERGENCE DYNAMICS: the author shows that the layer-wise ASRs converge to equilibrium, enabling the derivation of steady-state equations for linear layers

3.Training: MULTI-STAGED TRAINING PIPELINE: A series of work to simplify the model and improve its energy efficiency
-step1: KNOWLEDGE DISTILLATION from Non-spiking model (UniVTG)
-step2: supervised Finetune on the tasks
-step3: REPLACING SOFTMAX AND REMOVING LAYER NORMALIZATION
-step4: 1-BIT WEIGHT QUANTIZED SPIKINGVTG

The proposed method achieves comparable performance on QVHighlights and Charades-STA.

### Strengths
There're a few strenghts:
1. As far as I know, this is the first paper using SNN for VTG tasks.
2. The author proposed SALIENCY FEEDBACK GATING to preprocess multi-modal information during input. The specific method is to use the features output by the model to calculate Cosine Similarity for the Text Features features, and weight the Video Features based on the results. According to the authors, this module improves performance and reduces overall neural activity.
3. During training, the author uses CONVERGENCE DYNAMICS at equilibrium to reduce BPTT's dependence on the intermediate calculation graph.
4. In terms of model efficiency, the author fully integrates existing methods, including model distillation, softmax removal, layernorm removal, 1-bit model quantization, etc. Experimental results show that these operations do not lead to significant performance degradation, but fully improve the model's energy efficiency and deployability on neuromorphic chips.

Overall, I think this paper reveals the possibility of using SNN for low-energy VTG.

### Weaknesses
A. Methodology
Although the authors integrated many methods and first revealed the feasibility of VTG with SNN, I find it difficult to see a clear research question and clear original contribution besides SALIENCY FEEDBACK GATING (less than one page). Specifically:

1.I think SFG is an interesting and original module that uses the model’s output as a gating masked video input to enhance the model’s attention to key video segments. But the author lacks more demonstration of its effectiveness, motivation and scalability, and the main explanation is less than one page. Some questions are as follows:
-Can the author perform a visualization similar to gradcam to show the effectiveness of gating? (https://arxiv.org/abs/1610.02391)
-Does feedback connections participate in gradient backpropagation?
-The symbols and meaning of Eq4 need to be more clearly explained. I can read it, but the meaning of many symbols is not clearly pointed out.
-When calculating the dot product, how to ensure that the text feature M encoded by a single layer network and the video feature a encoded by a deep network are in the same feature space and can produce meaningful similarity measurements?
-Does this part of the module also perform subsequent 1-bit quantization? That is, weight values ​​in {−1, 1} and activations in ∈ {−1, 0, 1}?


2.Lack of novelty:
Except for SFG, most of the methods in this article come from previous work, including: spiking transformer, ternary spiking model,  knowledge distillation, removing softmax and layer norm, and 1-bit quantization. 

Some research backgrounds are not well introduced, such as Softmax free (ANN/SNN) transformer, eg:
https://arxiv.org/html/2206.08898v2
https://arxiv.org/pdf/2409.15375

These integrations make sense, but I struggle to see original contributions beyond demonstrating SNN feasibility.

3.Lack of Experiment
The author only conducted experiments on two datasets, QVHighlights and Charades-STA. This is a small part of the baseline UniVTG. Can the author add more baselines for comparison? See
https://arxiv.org/abs/2307.16715

### Questions
See weakness:
-Can the author perform a visualization similar to gradcam to show the effectiveness of gating? (https://arxiv.org/abs/1610.02391)

-Does feedback connections participate in gradient backpropagation?

-The symbols and meaning of Eq4 need to be more clearly explained. I can read it, but the meaning of many symbols is not clearly pointed out.

-When calculating the dot product, how to ensure that the text feature M encoded by a single layer network and the video feature a encoded by a deep network are in the same feature space and can produce meaningful similarity measurements?

-Does this part of the module also perform subsequent 1-bit quantization? That is, weight values ​​in {−1, 1} and activations in ∈ {−1, 0, 1}?

-Can the author add more baselines for comparison?

More Questions:
I am quite impressed by the accuracy of the 1-bit model shown by the author in table 3. Combined with SNN, quantization technology is expected to significantly reduce the size, calculation amount and energy consumption of the model. But I'm curious why the model's performance doesn't drop significantly at 1bit without doing additional operations? The author seems to be quite close to the ann full-precision baseline of UniVTG+PT when using W in {−1, 1} and activations in {−1, 0, 1}? Does SNN promote 1-bit quantization here? More specifically:
-If the author trains a 1-bit ann of the same size using the same method and training pipeline, how does the performance compare with SNN?
-For table3, can a more comprehensive ablation experiment be performed on 1-bit to reveal more information? Is the high performance just because Bitnet's (https://arxiv.org/pdf/2310.11453) is good enough, or is there something in the paper that brings additional benefits?
-In addition, please include the important information of the size of the model in the paper and results, especially the comparison before and after ann-snn quantization.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SpikingVTG, the first spiking neural network (SNN) approach to Video Temporal Grounding (VTG). The work introduces several key points: (1) a saliency feedback gating mechanism that improves performance while reducing neural activity, (2) a multi-stage training pipeline enabling efficient model variants, and (3) normalization-free and quantized versions for resource-constrained deployment. The authors demonstrate competitive performance with state-of-the-art models while achieving significant improvements in energy efficiency.

### Strengths
- First implementation of Video Temporal Grounding (VTG) task.
- Significant energy efficiency improvements with spiked-based operations and 1-bit precision
- Comparable performance with the previous methods
- Easy understanding with straight-forward figures and equations

### Weaknesses
- Only two datasets are used (QVHighlights and Charades-STA). Authors can show one more dataset such as TACoS if it is available.
- Training overhead coming from knowledge distillation (Connected to Question #3)
- Lack of results comparison with efficient VTG papers (Connected to Question #4)

### Questions
1. Some previous SNN papers used ternary spikes [1,2] like SpikeVTG. I just wondered if there is any result about binary spikes to show the effectiveness of ternary spikes.
2. In previous spike-based Transformer architecture, there is no need to use the Softmax function due to binarized QKV. However, this work still uses the activation function in self-attention like ReLU. Could you explain why your work should use the activation function?
3. Knowledge distillation with the ANN model should cause a huge training overhead. I wonder if the overhead, such as training time or memory, has been analyzed.
4. There are some works about efficient algorithms for VTG. It would be better to compare with [3,4].

[1] Guo, Yufei, et al. "Ternary spike: Learning ternary spikes for spiking neural networks." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 38. No. 11. 2024.
[2] Xing, Xingrun, et al. "SpikeLM: Towards General Spike-Driven Language Modeling via Elastic Bi-Spiking Mechanisms." arXiv preprint arXiv:2406.03287 (2024).
[3] Liang, Renjie, et al. "Efficient temporal sentence grounding in videos with multi-teacher knowledge distillation." arXiv preprint arXiv:2308.03725 (2023).
[4] Liu, Ye, et al. "$ R^ 2$-Tuning: Efficient Image-to-Video Transfer Learning for Video Temporal Grounding." arXiv preprint arXiv:2404.00801 (2024).

### Soundness
2

### Presentation
3

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
This paper presents SpikingVTG, a new SNN architecture for Video Temporal Grounding (VTG). This paper proposes Saliency Feedback Gating (SFG) mechanism to improve the alignment of video segments with text queries, a multi-stage training pipeline that leverages knowledge distillation from a teacher model, and removing traditional neural network operations like softmax and layer normalization, thus enhancing its deployability on neuromorphic hardware.

### Strengths
1. Introducing a spiking neural network for video temporal grounding is original and can potentially bridge a gap in VTG with energy-efficient models. Using SNNs for temporal data aligns well with the temporal dynamics needed for VTG. 

2. The inclusion of a 1-bit quantized variant enhances the practicality of deploying the model on resource-limited devices.

### Weaknesses
1. The presentation covers various aspects of spiking and non-spiking VTG models but lacks clarity in sections describing critical architectural details. For example, the specific spiking transformer backbone, video feature extraction method, and the text encoder used. It remains unclear whether the model performance significantly relies on an ANN for video features (e.g., a large foundational ANN model - CLIP's visual encoder).  How essential are the ANN features in terms of obtaining you final results? 

2. **Limited Ablation Studies**: The paper claims significant contributions from the SFG mechanism, Normalization-Free transformer, and knowledge distillation but lacks ablation studies to isolate their effects. For instance, a comparison of the SFG mechanism against vanilla spiking transformers would better illustrate its impact on model performance and efficiency. Similarly, experiments exploring the effect of softmax and layer normalization in the spiking attention module, effect of ANN features obtained from vidual encoder, performance w/wo knowledge distillation, comparison between implicit differentiation vs. BPTT, would further support the claimed efficiency benefits.

### Questions
Refer to weaknesses.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This manuscript describes an SNN-based model for implementing Video Temporal Grounding. It proposes a baseline structure/model for an SNN transformer suitable for the task, which the authors claim is computationally expensive, and they consider alternatives more computationally sustainable: One based on model distillation from a DNN, then two further variants, one which is simplified by being rid of normalization layers, and another which is further quantized for hardware friendliness/efficiency.

### Strengths
In order of importance as I perceived it 

A saliency feeback gating (SFG) module is proposed operating somewhat similarly (namely similar complexity) to a transformer block that aims to gate the processing of low saliency parts of the input video stream, thereby reducing computation. This appears to be a novel contribution that did not exist in previous so far literature on VLMs (according to the author's claim).

A version of SNN-based VTG model that attains SoA performance.

### Weaknesses
Overall I found the paper a bit cryptic and laconic in its explanations. I think it can be improved by providing a bit more detail about the model structure and implementation of the system, as well as clarifying several ambiguous parts (see suggestions and questions below).

Several of the novelty factors (such as the removal of the normalization layers, or the use of ReLU instead of SoftMax) lack explanation, justification or intuition. Next to that the fact that there are not many datasets available to benchmark with, leaves one with the question whether these "optimizations" work in general (as opposed to being dataset specific).

The main top-performance claim comes from model distillation with an equally top-performance pretrained teacher model, which leaves me with a question whether the actual such claimed-contribution is to be claimed by the authors.

My assessment takes these factors into account, and remains a bit conservative given my limited background in VTG SoA work, but would be happy to elevate the score, upon constructive defense of the paper.

### Questions
It is not clear how distances [d_s, d_e] are representing, and therefore also how they are computed. Given that the clips of a video have finite length, should I assume that they measure #clips from the current clip that need to be marked f=1 and scored ? It is therefore also ambiguous for understanding what b_i represents.

Also not clear is if the fixed-length clips of a vide have overlapping segments or not?

What is 1-dimensional Non-Maximum Suppression?

You never introduced f~, s~, b~ in the main text.

u[t+delta] does not make much sense to me. If time is discrete, then delta can only take value in {0,1}, and therefore coincides with u[t+1]. I guess what you intend is to introduce an intermediate variable for Vmem before firing, but I don't comprehend why you relate it to time t, as opposed to using a separate name.

The claim that using spikes (even ternary) enhances -- by definition -- performance is superficial. It is not the case in BNNs, why should it be in SNNs ? And if you refer to efficiency, why is that when there is additionally state to maintain in memory?. As far as I understand the reason SNNs are considered efficient is not grounded on performance and not in absolute terms.

Near lines l-252,253 I believe you need to fix the dimensions or transposes. As they stand, it does not add up correctly.

If I look at equations in (4), i have the impression that there are more than O(L_v D) FP ops. More like O(DL_v + D^2 + L_v^2), which means the size of D vs L_v matters.

From the description I cannot really imagine how the decoder looks like. Pls provide a figure and more explanation(s). Also pls explain how s score is computed with an equation (so far I get the impression that it is the same as a*_i but for the output layer ?)

Around l-308, what is the relaction of act_i[t] with a*_i or a_i ?

It is also not clear if the training in section 3.3 is part of the contribution or other work in literature. To be honest if has been assumed (by me) but not been clear in the paper text how the training work. Namely I assume that you let the network reverberate for a set number of timesteps (T) and then you average the spikes across the timesteps before you use that as a signal to back-propagate. Am I correct?

It is not clear to me the rationale for removing the layer normalization, nor that it would work in all occasions (with different datasets). Unless it has been something that you discovered through trial and error could you elaborate more on the intuition and explanation?

How is ReLU meant to replace softmax? Softmax has many inputs and one output, while ReLU has 1 input and one output. Moreover as with the normalization what is the intuition for removing a probabilistic relative-weighting score and replacing it with an unbounded function that is based on absolute values? And how does that help stabilizing the model.

In l-399 you remark a scaling constant which is not clear to me how it comes to play and why is it used ?

While I find nice and impressive the performance of the SNN, the fact that it comes after distillation of a pretrained DNN, it make me wonder if the performance does belong to the SNN (and its design choices). So first of all the numbers in table 3 come after re-distilling each of those models or not? I would hope each of these models is distilled separately. Then in table 2 the comparison to the other DNNs other than the UniVTG are I would say irrelevant unless you re-distil spiking VTG after each of them. Can you do that for a few of them? (namely test different teachers).

Near l-483 you state that the metric  represents the proportion og active neurons per tstep averaged across all layers. Are they accumulated across timesteps too or ?

In the analysis of energy and power efficiency, first of all you do not discuss power. Second you are considering only encoder attention and linear layers, but not the decoder. Why ? Moreover it is not clear to me if in the MAC/ACC costs you only consider the arithmetic operation or you also take into account the memory transactions to fetch weights, and if you also consider the cost of the memory transactions for the neuron states (Vmem), which is special in the stateful SNN neurons. Can you elaborate more on that say in connection to BNNs which are also quantized and binary but have no (recurrent) state?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This work introduces the SpikingVTG model for Video Temporal Grounding (VTG) by incorporating spiking neural networks (SNNs). To improve performance and reduce neural activity, this work proposes the saliency feedback gating mechanism (SFG), which also allows training with implicit differentiation at equilibrium to avoid BPTT. To further enhance computation efficiency, a multi-stage training pipeline that utilizes knowledge distillation and architectural adjustments, is used to develop the Normalization Free (NF)-SpikingVTG model and the 1-bit NF-SpikingVTG model. Experiments on QVHighlights and Charades-STA demonstrate the advantages of SpikingVTG in terms of performance and energy efficiency.

### Strengths
1. Introducing SNNs to VTG is an interesting exploration, which leads to an energy-saving model for the field.
2. The saliency feedback gating mechanism effectively conveys dynamic information to filter the inputs, which not only leads to sparser neural activity, but also improves model performance. Furthermore, introducing feedback connections allows the model to be trained with implicit differentiation at equilibrium.
3. The multi-stage training pipeline incorporating knowledge distillation and architectural simplification develops models with higher computational efficiency and low degradation in performance.

### Weaknesses
Major points:
1. The main weakness is that the network cannot be considered a fully spiking neural network, and it's not clear if it could be deployed on existing neuromorphic chips. Although the computational complexity of SFG is much lower compared to spiking transformers, it is still impossible to ignore the floating-point multiplication operations and nonlinear computations (softmax), which are not in line with the characteristics of SNN. In addition, the gelu in the intermediate layer (line 729) and the floating-point communications through the residual connection in the output layer (Figure 1 and line 735) are also problematic. Therefore, the network is best called a hybrid SNN and I think the authors should change "Yes" to "Partial" in Table 1 and 2.
2. The technical value of this work seems to be mostly an engineering contribution, and I'm not sure about the value of the innovation, since many key methods come from other work, such as training with implicit differentiation at equilibrium [1], replacing softmax with relu [2], and 1-bit quantization [3].
3. Given that the network is a hybrid SNN, it performs only marginally better than SpikeMba, which is also a hybrid SNN, on some metrics, and even worse on others (Table 1). Besides, there is a lack of performance comparison between the two proposed variants and non-spiking models.

Minor points:
1. In addition to comparing performance, it would be better to report the inference time and the number of model parameters to provide more evidence of the computational efficiency.
2. The numbering in section 4.3.1, "Analysis of Energy and Power Efficiency", is incorrect. It should be at the same level as section 4.3, "Ablation Study", but not a subsection of it.
3. Numerous equations are mixed in with the text, making them less prominent. This is particularly evident in the appendix section on the loss function, where the total loss function consists of three separate loss functions. The formulas for the first two loss functions are presented separately, but the formula for the last term $L_c$ is mixed in with the text, which may cause inconvenience when reading.

[1] Mingqing Xiao, Qingyan Meng, Zongpeng Zhang, Yisen Wang, and Zhouchen Lin. Training feedback spiking neural networks by implicit differentiation on the equilibrium state. Advances in Neural Information Processing Systems. 2021.

[2] Kai Shen, Junliang Guo, Xu Tan, Siliang Tang, Rui Wang, and Jiang Bian. A study on relu and softmax in transformer. arXiv. 2023.

[3] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models. arXiv. 2023.

### Questions
1. How do the authors remove the layer normalization in section 3.4.2? Is it simply removed or are other adjustments introduced? It would be good to include details of the implementation in the text.

### Soundness
3

### Presentation
3

### Contribution
2

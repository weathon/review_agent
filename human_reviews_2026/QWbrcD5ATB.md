# Sample-efficient Integration of New Modalities into Large Language Models

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 6, 4, 4

## Abstract
Multimodal foundation models can process several modalities. However, since the space of possible modalities is large and evolving over time, training a model from scratch to encompass all modalities is unfeasible. Moreover, integrating a modality into a pre-existing foundation model currently requires a significant amount of paired data, which is often not available for low-resource modalities. In this paper, we introduce a method for sample-efficient modality integration (SEMI) into Large Language Models (LLMs). To this end, we devise a hypernetwork that can adapt a shared projector—placed between modality-specific encoders and an LLM decoder—to any modality. The hypernetwork, trained on high-resource modalities (i.e., text, speech, audio, video), is conditioned on a few samples from any arbitrary modality at inference time to generate a suitable adapter. To increase the diversity of training modalities, we artificially augment the number of encoders through isometric transformations. We find that SEMI achieves a significant boost in sample efficiency during few-shot integration of new modalities (i.e., satellite images, astronomical images, inertial measurements, and molecules) with encoders of arbitrary embedding dimensionality. For instance, to reach the same accuracy as 32-shot SEMI, training the projector from scratch needs 64$\times$ more data. As a result, SEMI holds promise to extend the modality coverage of foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes SEMI (Sample-Efficient Modality Integration), a three-stage framework to integrate new modalities into LLMs. The method leverages a hypernetwork trained on high-resource modalities to generate low-rank adapters for a shared MLP projector, enabling adaptation to unseen low-resource modalities such as satellite images, astronomical data, IMU signals, and molecular structures. The authors introduce a new benchmark and demonstrate that SEMI achieves competitive performance with significantly fewer samples compared to baselines.

### Strengths
1. This paper introduces a new dataset CAPDELS, which is a new testbed for related works.

2. The experiment results outperform several baselines, providing valuable empirical reference.

### Weaknesses
1. The core idea of using a two-component architecture has been explored in prior works on multimodal adaptation. While SEMI introduces a hypernetwork to generate adapters, the paper does not sufficiently clarify how this differs from fine-tuning or lora-tuning LLM parameters. A more thorough discussion would strengthen the novelty claim.

2. The proposed hypernetwork introduces a non-negligible number of additional trainable parameters during the second training stage. In contrast, baselines like LoRA or direct projector fine-tuning operate with fewer or no extra parameters. This might makes an unfair comparison.

3. The evaluation is exclusively conducted on captioning tasks across all modalities. While captioning is a common benchmark, it represents only one type of multimodal integration. Given that the paper proposes a general framework for LLMs' modality integration, broader task coverage would be necessary to substantiate this claim.

### Questions
See `Weakness' section.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper focuses on how to integrate new modality into pre-existing foundation models. A method for sample-efficient modality integration is proposed, namely SEMI. A hyper-network is proposed that adapt a shared projector--placed between modality-specific encoders and a LLM decoder--to any modality. SEMI achieves performance boost in sample efficiency during few-shot integration of new modalities.

### Strengths
1. It is important to achieve integrating new modality into a pre-existing foundation model.

2. Method is clearly illustrated throw simple figure and detailed text.

### Weaknesses
1. What's the differences between the proposed method against the Bind-style frameworks?

[A] Girdhar R, El-Nouby A, Liu Z, et al. Imagebind: One embedding space to bind them all[C]//Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2023: 15180-15190.

[B] Lyu Y, Zheng X, Zhou J, et al. Unibind: Llm-augmented unified and balanced representation space to bind them all[C]//Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024: 26752-26762.

[C] Zhu B, Lin B, Ning M, et al. LanguageBind: Extending Video-Language Pretraining to N-modality by Language-based Semantic Alignment[C]//The Twelfth International Conference on Learning Representations.

It would be better to clearly discuss this line of research with SEMI.

2. The experimental results in the main paper are limited. The reviewer totally understand there are more in the appendix, however, the paper needs better arrangement of layout and content distribution.

### Questions
## Relation to Bind-style Frameworks (ImageBind, UniBind, LanguageBind)

The current manuscript does not clearly situate the proposed method within the broader line of Bind-style multi-modal frameworks such as ImageBind [Girdhar et al., CVPR 2023], UniBind [Lyu et al., CVPR 2024], and LanguageBind [Zhu et al., ICLR 2024]. These works also target unified and sample-efficient representation learning. A deeper comparison would strengthen the contribution.


### Question 1: 
How does the proposed method differ conceptually from the Bind-style frameworks that unify representations across modalities?

### Question 2: 
Have the authors considered incorporating or comparing SEMI with Bind-style pretrained embeddings (e.g., using ImageBind features as state encoders)? A brief empirical or conceptual comparison could clarify whether SEMI complements or diverges from representation-binding paradigms.

### Question 3: 
Can SEMI be regarded as a task-level integration framework (fusing learning paradigms) while Bind-style frameworks operate at the representation-level? If so, could combining the two approaches further enhance sample efficiency or generalization? This discussion would be beneficial to include in the related work or discussion section.

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
3

### Summary
The paper studied a sample-efficient approach to integrating new modalities (with low resources) into large language models by training a hypernetwork that can adapt a shared projector and isometric transformation of encoder output. Also, the authors contributed to preparing benchmarks and creating a new dataset. The overall writing and organization of the paper are good.

### Strengths
1) The background, motivation, and problem statement are well described and justified. It is a meaningful and valuable problem to be addressed.
2) The proposed method is simple and straightforward, and does not require complicated theory and modules.
3) The experiment aligns with the objective of the work, supporting the claim of sample efficiency.

### Weaknesses
1) In line 111, modality-specific encoders are assumed. In many real-world problems, the encoders might not be available. This assumption is quite restrictive.
2) The proposed method is compared with weak baselines. While authors explained that they might require large-scale paired data, a more comprehensive comparative analysis can be provided to show the tradeoff between data efficiency and performance. To show that this work extends the SOTA in a certain way. 
3) The number of new modalities studied is limited, and the generalization and scalability of the work are unclear.
4) The proposed method is only assessed on integrating one extra modality each time and does not evaluate its performance when multiple modalities are simultaneously used for cross-modal synergy.

### Questions
1) The proposed method just appears to be a better way of initializing the projector? Hence, better than weak baselines?
2) What is the performance of the method? If the third stage is ablated.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes SEMI, a sample-efficient modality integration method. Instead of retraining a new projector for each modality, SEMI employs a hypernetwork trained on multiple high-resource modalities (e.g., image, audio, video) to learn how to generate LoRA adapters for a shared projector. During adaptation to a new, low-resource modality (e.g., molecules), the trained hypernetwork observes only a few paired examples to produce a modality-specific adapter, which is then further fine-tuned on the same few-shot set to integrate the new modality into the LLM.
To enhance generalization, the authors introduce isometric transformations to artificially diversify encoder representations and apply text grounding to align modality embeddings with textual semantics. Experiments show that SEMI achieves strong sample efficiency. For instance, with only 32 examples, it matches the performance of baseline methods (e.g., training the projector or LoRA adapter from scratch) that require 16× even 64× more data.

### Strengths
Strengths:

1.The paper targets a practical yet underexplored problem. As AI increasingly extends into scientific and industrial domains, this work provides a solution for efficiently integrating new modalities into frozen large language models.

2.On several unseen modalities, particularly those that differ significantly from training modalities, such as IMU signals and molecular structures, SEMI demonstrates clear advantages in the low-data regime.

3.The evaluation spans a diverse set of modalities, supporting the method’s data efficiency and generalization robustness.

### Weaknesses
1.Stage 3 consists of two steps: (1) the hypernetwork generates a LoRA adapter ; and (2)  is then fine-tuned on the same K-shot samples. The paper lacks a critical ablation: what happens if the generated adapter is used without fine-tuning (i.e., zero-shot adaptation)? If performance remains strong, the hypernetwork truly learns to generate; if not, it may merely provide a better initialization while fine-tuning does most of the adaptation. The current setup mixes these effects.

2.Current experiments focus solely on captioning tasks, with no assessment of semantic understanding or reasoning capabilities.

3.The work lacks comparisons to cross-attention or Q-Former–style multimodal adapters, which are the most direct architectural competitors.

4.The paper discusses the hypernetwork’s training challenges only qualitatively, without reporting runtime or resource usage. Moreover, the current projector is a two-layer MLP, and the hypernetwork generates LoRA parameters for its first layer. For more complex projectors (e.g., Q-Former), the generation burden could grow rapidly, leading to scalability and optimization issues. The authors mention training challenges only qualitatively, without a deeper discussion of scalability limitations.

The claim that random orthogonal matrices “emulate numerous encoders” is overstated. Geometrically, such transformations (e.g., rotations) preserve inner products and pairwise distances, merely changing the coordinate system rather than altering the data distribution or encoder behavior. They act more as a robustness augmentation than as true encoder simulation.

### Questions
See weaknesses

### Soundness
2

### Presentation
2

### Contribution
2

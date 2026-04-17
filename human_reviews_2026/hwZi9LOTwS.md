# Spatial-Temporal-Spectral Unified Modeling for Remote Sensing Dense Prediction

- Decision: Reject
- Scores: 8, 4, 2, 2

## Abstract
The proliferation of multi-source remote sensing data has propelled the development of deep learning for dense prediction, yet significant challenges in data and task unification persist. Current deep learning architectures for remote sensing are fundamentally rigid. They are engineered for fixed input-output configurations, restricting their adaptability to the heterogeneous spatial, temporal, and spectral dimensions inherent in real-world data. Furthermore, these models fail to leverage the intrinsic correlations across different remote sensing dense prediction tasks, necessitating the development of distinct models or task-specific decoders. This paradigm is also limited to a fixed set of output semantic classes that must be learned during training, where any change to the classes requires costly retraining. To overcome these limitations, we introduce the Spatial-Temporal-Spectral Unified Network (STSUN) for unified modeling. STSUN can adapt to input and output data with arbitrary spatial sizes, temporal lengths, and spectral bands by leveraging their metadata for a unified representation. Moreover, STSUN unifies disparate dense prediction tasks within a single architecture by conditioning the model on trainable task embeddings. STSUN enables flexible prediction across multiple sets of semantic categories by integrating trainable category embeddings as metadata. Extensive experiments on multiple datasets with diverse Spatial-Temporal-Spectral configurations in multiple scenarios demonstrate that a single STSUN model effectively adapts to heterogeneous inputs and outputs, unifying various dense prediction tasks and diverse semantic class predictions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces a novel framework for remote sensing dense prediction. The core contribution is a unified model capable of handling heterogeneous inputs and outputs across spatial, temporal, and spectral dimensions. Furthermore, the model unifies three distinct dense prediction tasks and supports flexible semantic categories, therefore achieving data-task-class unification. The authors conduct experiments on seven diverse datasets of building and land use/land cover scenarios, demonstrating that their single unified model not only adapts to varied data-task-class configurations but also achieves state-of-the-art performance across all of them.

### Strengths
1.	The primary strength lies in its novel approach to unification. While prior work has used hypernetworks or other approaches for adaptive unification, it was typically limited to the channel dimension to handle multi-modal data [1-4]. This work innovatively connects the diversity across spatial, temporal, and spectral dimensions with the non-uniformity challenges in remote sensing concerning data, tasks, and classes. By unifying these core dimensions via a hypernetwork, the proposed method provides a natural and effective solution for a unified data-task-class framework.
2.	The decision to treat the temporal dimension separately, acknowledging its unique characteristics and designing a specific unification module for it, is an interesting and convincing design choice.
3.	The proposed method for unifying the spatial, temporal, and spectral dimensions of inputs and outputs appears to be general and highly adaptable, suggesting potential applicability to other research domains.
4.	The paper is supported by comprehensive and convincing experimental results. In particular, the ablation studies on multi-task and multi-class unification demonstrate the necessity and benefits of the proposed unified approach.
5.	The paper is well-written, clearly structured, and easy to follow.

[1] Xiong, Z., Wang, Y., Zhang, F., Stewart, A. J., Hanna, J., Borth, D., ... & Zhu, X. X. (2024). Neural plasticity-inspired multimodal foundation model for earth observation. arXiv preprint arXiv:2403.15356.
[2] Li, X., Li, C., Ghamisi, P., & Hong, D. (2025). Fleximo: A flexible remote sensing foundation model. arXiv preprint arXiv:2503.23844.
[3] Zhang, Y., Li, W., Zhang, M., Han, J., Tao, R., & Liang, S. (2025). SpectralX: Parameter-efficient domain generalization for spectral remote sensing foundation models. arXiv preprint arXiv:2508.01731.
[4] Sumbul, G., Xu, C., Dalsasso, E., & Tuia, D. (2025). SMARTIES: Spectrum-Aware Multi-Sensor Auto-Encoder for Remote Sensing Images. arXiv preprint arXiv:2506.19585.

### Weaknesses
1.	Some implementation details are not fully clear.
2.	The unification of each dimension is formalized with notation and some equations. However, the presentation could be made even clearer and more rigorous with a more comprehensive mathematical formulation.
3.	Moving some of the key visualization results from the appendix to the main paper would make the results more compelling and easier for the reader to interpret.
4.	It is recommended to tone down claims like "for the first time" in the introduction to avoid potential disputes and strengthen the paper's scholarly tone.

### Questions
1.	The proposed model not only achieves data-task-class unification but also outperforms the baseline models. What are the factors contributing to this superior performance?
2.	In Appendix A.1, the hypernetwork generates adaptive weights and biases for some dimensions, but only adaptive weights for others. What is the reasoning behind this design choice?
3.	The model requires explicit metadata as input. How could this metadata be inferred implicitly in future work to make the model more streamlined and user-friendly?
4.	This work primarily focuses on supervised models. What role could this unification framework play when extended to vision foundation models or large multi-modal models?

Minor Comments
1.	In Figure 2, consider adding "×N" to the encoder and decoder blocks to indicate that they are repeated.
2.	In Figure 3, the text should be ordered from top to bottom to maintain consistency with the flow in Figure 2.
3.	In Table 1, it would be helpful to include the size (e.g., number of samples) for each dataset.
4.	In Table 1, "Image Size" should be changed to "(H, W)" to be consistent with the dimensional notation used in the text.
5.	For all tables reporting results, consider adding arrows next to each metric to improve clarity, similar to the presentation in Table 8.
6.	The layout of some equations in Figure 5 needs adjustment for better readability.
7.	The positioning of Figure 6 and Figure 7 could be modified.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper presents a unified framework addressing the challenge of inconsistent input and output configurations across different dense-prediction tasks in remote sensing. The proposed Spatial-Temporal-Spectral Unified Network (STSUN) integrates two key components: the Dimension Unified Module (DUM), which employs a transformer-based hypernetwork conditioned on metadata to adaptively map variable dimensions, and the Local-Global Window Attention (LGWA) module, which captures multi-scale contextual relationships. The model is designed to handle multiple dense-prediction tasks, including semantic segmentation, binary change detection, and semantic change detection, and can be trained in either single-task or multi-task settings, flexibly adapting to different spatial, temporal, and spectral domains.

### Strengths
The paper addresses an important and practical issue in remote sensing: heterogeneity of input and output structures. Encoding spatial, temporal, and spectral configurations as metadata is a smart, scalable idea. The local-global attention design helps handle multi-resolution dependencies, and multi-task training improves performance. Experiments are extensive, and results are strong across benchmarks.

### Weaknesses
1. Despite the strong empirical results, the core components are based on existing ideas. The DUM is a direct application of a transformer-based hypernetwork, and the LGWA is conceptually similar to other multi-scale windowed attention mechanisms found in models like the Swin Transformer or SegFormer.
2. The paper lacks theoretical analysis or deeper insight into its decoupled unification strategy, relying solely on experimental ablation (Table 11) to justify its design.
3. Although the appendix includes implementation details, the main exposition can be conceptually unclear. Notation is inconsistent, metadata definitions are vague, and the link between input and output dimensions must be inferred. These issues hinder a full understanding and make reimplementation challenging. For example, the notational system ($T_1$, $T_2$, $C_1$, $C_2$) is confusing, and metadata ($M_{in}$, $M_{out}$) lacks clear definition. Please clarify these structures and their correspondence.
4. The description of data dimensions (on page 1) and Figure 2 is overly verbose and could be condensed for readability.

### Questions
1. The "flexible category set" capability (Section 4.4) relies on selecting from a predefined and trainable set of class embeddings. Could the authors clarify the model's behavior with a truly 'new' or 'unseen' category not included in this predefined set? Would this scenario require retraining to add a new embedding, or can the model generalize in a zero-shot manner?
2. The main comparison tables (e.g., Tables 2-8) compare the STSUN_unified model (trained on combined data) against SOTA methods trained on single datasets. This makes it difficult to distinguish architectural benefits from the benefits of multi-task/multi-dataset training. Could the authors add the STSUN_single results (from Table 9) to these main tables for a more direct comparison against the SOTA baselines?
3. Please provide parameter counts or FLOPs for the STSUN model and the key baselines. This would help clarify whether the performance gains stem from the proposed architecture or from a significantly larger model capacity.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This manuscript proposes the Spatial-Temporal-Spectral Unified Network (STSUN), a framework designed to achieve unified dense prediction across diverse remote sensing tasks and data configurations. The authors identify key limitations in existing deep learning models for remote sensing, including fixed input-output configurations, task-specific architectures, and rigid category sets, which hinder adaptability to heterogeneous data and multi-task scenarios.

### Strengths
1.	The paper is clearly written and easy to follow, with a well-structured presentation of the problem and proposed approach.
2.	The idea of unifying spatial, temporal, and spectral dimensions within a single framework is interesting and relevant to challenges in remote sensing dense prediction.

### Weaknesses
1.	The unification of spatial, temporal, and spectral dimensions has already been explored in several recent remote sensing foundation models, such as RingMo-Agent [1] and Falcon [2], which aim to build unified representations across multi-platform and multi-modal data. The paper does not discuss or compare its approach with these existing large-scale models, limiting the clarity of its novelty and positioning. [1] RingMo-Agent: A Unified Remote Sensing Foundation Model for Multi-Platform and Multi-Modal Reasoning [2] Falcon: A Remote Sensing Vision-Language Foundation Model
2.	The idea of flexible semantic class sets is not entirely novel. Prior works on open-vocabulary and open-set segmentation in remote sensing already address similar challenges. The paper does not discuss or position its proposed trainable category-embedding mechanism relative to these approaches, which reduces the clarity of its contribution in this context.
3.	The method relies on metadata and hypernetworks to generate adaptive linear layers for unifying arbitrary spatial, temporal, and spectral dimensions. In practice, the claimed “complete unification of STS dimensions” may be constrained by variations in spatial resolution, spectral coverage, and temporal sampling intervals. The authors do not provide sufficient experimental validation to support the generalizability of this approach.
4.	The feasibility of the Temporal Unified Module (TUM) is unclear. TUM fuses multi-temporal features using hypernetworks and metadata, mapping them to arbitrary output temporal lengths. For high-temporal-resolution change detection or long sequence data, such linear mappings may not adequately capture complex temporal dynamics, potentially causing information loss or performance degradation. The paper does not include experiments on long temporal sequences or ablation studies to validate TUM’s effectiveness.
5.	The Local-Global Window Attention (LGWA) module uses multiple local windows of predefined shapes alongside a single global window to capture features at different scales. Fixed window sizes and shapes may not adapt well to varying spatial resolutions or object scales. When input data vary substantially, for example in satellite type, spatial resolution, or spectral channels, this strategy may lead to unstable performance. Furthermore, no experiments are provided comparing LGWA with other adaptive attention mechanisms such as Swin Transformer or CSWin.
6.	The experimental results show that the proposed method’s performance is not particularly strong. Compared with existing state-of-the-art methods for building extraction and building change detection, the accuracy exhibits a noticeable gap. Additionally, the paper does not include comparisons with large foundation models such as the Segment Anything Model or remote sensing models based on SAM, which would help contextualize the method’s practical effectiveness.
7.	The ablation studies and analysis are limited, making it difficult to fully support the authors' claim of achieving unification across arbitrary spatial, temporal, and spectral dimensions. More comprehensive experiments are needed to demonstrate the contribution of each component and to validate the generalization of the proposed framework.

### Questions
1.	Relation to existing foundation models: Could the authors clarify how STSUN differs from recent remote sensing foundation models such as RingMo-Agent and Falcon? Have the authors considered including a comparison or discussion of these models to better position the novelty of their approach?
2.	Flexible semantic class sets: How does the proposed trainable category-embedding mechanism compare with prior open-vocabulary or open-set segmentation approaches in remote sensing? Could the authors provide experiments or analysis to demonstrate the advantage of their method over these existing paradigms?
3.	STS dimension unification: The method relies on metadata and hypernetworks to unify spatial, temporal, and spectral dimensions. Can the authors provide more empirical evidence to show that this approach generalizes across varying spatial resolutions, spectral coverage, and temporal sampling intervals? For example, have they tested the model on datasets with highly heterogeneous input configurations?
4.	Temporal Unified Module (TUM): For long temporal sequences or high-temporal-resolution change detection, how does TUM handle complex temporal dynamics? Could the authors include ablation studies or experiments on longer sequences to validate the effectiveness and stability of TUM?
5.	Local-Global Window Attention (LGWA): How sensitive is LGWA to the choice of local window sizes and shapes, particularly when input data vary in spatial resolution, object scale, or spectral channels? Have the authors compared LGWA with other adaptive attention mechanisms such as Swin Transformer or CSWin to verify its effectiveness?
6.	Experimental performance and comparisons: The current experiments show a noticeable gap in accuracy compared with state-of-the-art building extraction and change detection methods. Could the authors provide comparisons with foundation models such as the Segment Anything Model or RS models based on SAM to better contextualize the performance?
7.	Ablation studies: The current ablation experiments appear limited. Could the authors provide more detailed component-level analyses to demonstrate the contribution of each module and to support their claim of achieving full unification across arbitrary spatial, temporal, and spectral dimensions?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The authors propose the Spatial-Temporal-Spectral Unified Network (STSUN) to leverage diverse dimensions of input data for dense downstream tasks. The proposed method uses metadata to learn representations from diverse spatial resolutions, temporal lengths, and spectral bands.

The overall framework aims to integrate diverse dense tasks within a single architecture using task embeddings that support arbitrary class subsets. It is composed of fives proposed modules: the Input Spatial-Spectral Unified Module (ISSUM) extracts spatial-spectral embeddings, the Local-Global Window Attention mechanism (LGWA) learns multi-spatial patterns, the Temporal Unified Module (TUM) homogenizes the temporal dimension, the Decoder Local-Global Attention Blocks processes the global embedding and the Output Spatial-Spectral Unified Module (OSSUM) generates the outputs.

The authors conducted exhaustive experiments on building and land use land cover (LULC) scenarios, comparing an extensive list of competing methods, including a version of their model trained on a single dataset and ablation studies to partially justify their approach.

### Strengths
I acknowledge that this review has been produced considering the revised manuscript.

1. The framework, which incorporates metadata of data inputs and dense prediction tasks, is well introduced and carefully distinguishes the spatial, temporal, and spectral input and output dimensions.

2. The authors leveraged metadata to generate an adapted architecture for input and output dimensions, proposing five modules to unify their representations.

3. The learnable task embedding effectively guides the model to perform a given task within the set of possible dense tasks.

4. The category embedding set enables flexible predictions across predefined semantic categories.

5. The experiments are comprehensive, including a relevant list of competing methods specific to each dataset and scenario, an ablation study of several proposed modules, and an analysis of single-dataset versus multi-dataset training of their method.

### Weaknesses
I acknowledge that this review has been produced considering the revised manuscript.

1. The limitations that this work addresses (L.90-103) are partially studied and grounded; however, several nuances should be considered, which weaken the motivations:  
 a. Fixed configurations: ViTs are capable of processing sequences of arbitrary length; if spatial-temporal-spectral cubes are divided into tokens, this is no longer a limitation in theory [1, 2, 3, 4, 5].  
 b. Fixed task: I would like to quote a recent work by Siméoni et al. [6], which aligns with recent remote sensing literature: "In particular, SSL produces rich, high-quality visual features that are not biased toward any specific supervision or task, thereby providing a versatile foundation for a wide range of downstream applications." Since most remote sensing foundation models and generalist backbones achieving state-of-the-art performance align with such statements, one may question whether learning correlations between the mentioned tasks is necessary, or experimental comparisons against SSL-pretrained models should be provided to support this argument.  
 c. Fixed set of categories: Remote sensing foundation models and generalist backbones are designed to be adapted to dense tasks with a single linear layer, which may be simpler than the authors' proposed method (see W.4.c). For example, species distribution models, well established in remote sensing, produce a distribution of classes per pixel, ranging up to 10,000 classes, making these approaches nearly agnostic to additional classes [7].  

2. There is insufficient related work on spatial-temporal-spectral methods for remote sensing pretraining and applications, such as S2MAE [2], SkySense++ [4], Galileo [5], all demonstrating strong performance on dense tasks with task-specific heads without requiring cross-task correlation training. Additionally, the differences between the proposed method and TSViT [7] are unclear, as their method efficiently processes spectral, spatial, and temporal information in a similar sequence, as illustrated in Figure 2 and explained in Section 3.2. A comprehensive comparison of the proposed method with existing remote sensing domain methods would be appreciated.

3. Despite the generalist formulation and architectural adaptability through hypernetworks to create adapted linear layers, once the model is instantiated with fixed dimensions (Eq. 3), it cannot accommodate different dimensions, such as shorter time series or different numbers of spectral bands, due to the fixed size of linear layers used for projections (Figure 6). Although the core transformations operate within unified dimensions, linear layers must still be adapted for each module to change dimensions across datasets, which could be reframed as modality-specific or task-specific tokenizers, as used in remote sensing foundation models.

4. It is unclear why foundation models, essential methods in remote sensing nowadays, were excluded from this study despite the authors' statements in A.6. Remote sensing foundation models are not restricted to unifying input representations, as mentioned in A.6; they aim to learn generalist representations that perform better on any downstream task, including all dense prediction tasks and many others (see the quote from Siméoni et al. in W.1.b).  
a. The authors state that their method aims to better unify input and output dimensions, which could be interpreted as learning generalist representations that map any input and output dimensions. This is precisely the purpose of remote sensing foundation models, and thus both approaches address the same research direction.   
b. The authors also state that "STSUN is not designed to compete with the massive parameter space of models like Falcon"; however, they compared their method to models with parameter counts ranging from 6.26M to 468.25M (Table 12). Most remote sensing foundation models operate in the same range by proposing multiple backbone versions, e.g., Galileo (0.8M to 85M) [5], OlmoEarth (1.4M to 300M) [9], AnySat (125M to 128M) [10], S2MAE (86M to 632M) [2], and others [1, 3, 11] that would be suitable for fair comparison.  
c. Foundation models are designed to be either fully fine-tuned or fine-tuned with a single linear layer to perform any dense task. Since the proposed OSSUM requires two linear layers and a transformer block to project the class dimension, plus an additional layer to project to the output space, comparison with foundation models should be considered fair.

5. There is insufficient comparison with Perceiver IO [12], which demonstrates a very similar method based on attention fusion to unify input dimensions through linear transformations and learns the global output tensor carrying task information, which is generated with an output query.  

6. The proposed Local-Global Window mechanism is not novel, as it was introduced in Swin Transformers [13].  

7. There is no LULC scenario with time series, despite this being one of the most common use cases, particularly for agricultural applications. The PASTIS [14] dataset would be a well-suited benchmark for this study.  

8. STSUN appears to be the only model trained on six datasets, whereas competing methods appear to be trained only on the dataset of interest, making direct comparison unfair. Training a similar method such as TSViT [8] or Perceiver IO [12] on multiple datasets would better assess whether performance gains result from the proposed architecture or from the combination of datasets used in training. Note that fine-tuning a foundation model on each dataset would be more appropriate, as they have been pretrained on larger datasets than competing methods.

9. The experiments lack standard error estimates evaluating the variability of the proposed method (e.g., across different initialization seeds), which raises questions about the significance of numerically similar results. For example, it is difficult to conclude that the "category unification" strategy is relevant based on Table 10 results, given the very close numerical values that could fall within the same standard error range.

10. There is a lack of ablation study to justify the selection of all proposed modules.

11. The overall paper is difficult to read due to the introduction of numerous proposed concepts, some of which are not novel, and a notation framework that is initially well introduced but subsequently difficult to follow because of exhaustive naming of modules and layers that is not necessary. Figure 6 should be improved and referenced correctly to help readers understand each module and how they interact. Note that several layer and module formulations are redundant; the submission would benefit from consolidating them into a more generalist concept.

## References:

[1] Xiong et al., Neural Plasticity-Inspired Multimodal Foundation Model for Earth Observation. In ArXiv 2024.

[2] Li et al., S2MAE: A Spatial-Spectral Pretraining Foundation Model for Spectral Remote Sensing Data. In CVPR 2024.

[3] N. Bountos et al., FoMo: Multi-Modal, Multi-Scale and Multi-Task Remote Sensing Foundation Models for Forest Monitoring. In AAAI 2025.

[4] K. Wu et al., A semantic-enhanced multi-modal remote sensing foundation model for Earth observation. In Nature machine intelligence 2025.

[5] G. Tseng et al., Galileo: Learning Global & Local Features of Many Remote Sensing Modalities. In ICML 2025.

[6] Siméoni et al., DinoV3. In ArXiv 2025.

[7] Zbinden et al., MaskSDM: Adaptive Species Distribution Modeling Through Data Masking. In ECCV Workshop.

[8] Tarasiou et al., ViTs for SITS: Vision Transformers for Satellite Image Time Series. In CVPR 2023.

[9] Herzog et al., OlmoEarth: Stable Latent Image Modeling for Multimodal Earth Observation. In ArXiv 2025.

[10] G. Astruc et al., AnySat: One Earth Observation Model for Many Resolutions, Scales, and Modalities. In CVPR 2025.

[11] A. Fuller et al., CROMA: Remote Sensing Representations with Contrastive Radar-Optical Masked Autoencoders. In NeurIPS 2023.

[12] Jeagle et al., Perceiver IO: A General Architecture for Structured Inputs & Outputs. In ICLR 2022.

[13] Liu et al., Swin Transformer: Hierarchical Vision Transformer using Shifted Windows. ICCV 2021.

[14] Sainte Fare Garnot et al., Panoptic Segmentation of Satellite Image Time Series with Convolutional Temporal Attention Networks. In ICCV 2021.

### Questions
## Questions
1. Considering the details provided in W.1, to what extent does unifying dense tasks provide benefits compared to using task-specific, parameter-efficient heads? One would expect experimental evidence demonstrating that learning task correlations improves results compared to existing methods.

2. Considering the details provided in W.1 and W.4, to what extent does a fixed set of classes introduce rigidity in the generalization of large backbones, given that a simple linear layer can be fine-tuned for the final task? One would expect experimental evidence demonstrating that using category embedding sets improves results compared to a plug-and-play approach.

3. What are the key differences between the proposed approach and a foundation model with one linear layer per dataset to perform each dense task simultaneously?

4. Since the proposed method depends on input and output dimensions (Section 3.1), do all linear layers change with respect to each dataset used in training? If so, is it correct that one must train dataset-specific linear layers?

5. What are the differences between the proposed method and Perceiver IO [12]? Why was Perceiver IO not considered as a competing method, given that its overall goal is to unify any input dimensions and output tasks?

6. How are the category embedding set and selected category subsets defined? Is the method still limited to a fixed number of categories, as described in the limitations?

References are listed in the Weaknesses section.

## Comments

- Section 3.2 lacks references to the architecture details of each module (Appendix A.1). This section is important to understand the methodology while being difficult to read and follow all the steps.
- Figure 6: this crucial figure is confusing; it lacks links between the modules to better understand how dimensions match between each module.
- Figure 6: to what corresponds to the list of numbers related to the spectral wavelength input in (a)?
- Table 1: please explain all acronyms in the caption.

### Soundness
1

### Presentation
1

### Contribution
2

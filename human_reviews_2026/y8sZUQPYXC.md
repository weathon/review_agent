# PartSAM: A Scalable Promptable Part Segmentation Model Trained on Native 3D Data

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 6

## Abstract
Segmenting 3D objects into parts is a long-standing challenge in computer vision. To overcome taxonomy constraints and generalize to unseen 3D objects, recent works turn to open-world part segmentation. These approaches typically transfer supervision from 2D foundation models, such as SAM, by lifting multi-view masks into 3D. However, this indirect paradigm fails to capture intrinsic geometry, leading to surface-only understanding, uncontrolled decomposition, and limited generalization. We present PartSAM, the first promptable part segmentation model trained natively on large-scale 3D data. Following the design philosophy of SAM, PartSAM employs an encoder–decoder architecture in which a triplane-based dual-branch encoder produces spatially structured tokens for scalable part-aware representation learning. To enable large-scale supervision, we further introduce a model-in-the-loop annotation pipeline that curates over five million 3D shape–part pairs from online assets, providing diverse and fine-grained labels. This combination of scalable architecture and diverse 3D data yields emergent open-world capabilities: with a single prompt, PartSAM achieves highly accurate part identification, and in a “Segment-Every-Part” mode, it automatically decomposes shapes into both surface and internal structures. Extensive experiments show that PartSAM outperforms state-of-the-art methods by large margins across multiple benchmarks, marking a decisive step toward foundation models for 3D part understanding.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose PartSAM, a model for 3D part segmentation. Specifically, the PartSAM contains two essential components: encoder with initialized weight from previous work that inherits 2D foundation prior, and further train with native 3D data; and scaling the 3D data with model-in-the-loop annotation. Then the scaling of data results in a very powerful model for 3D part segmentation, with great in-the-wild generalization ability, and better performance comparing to previous methods.

### Strengths
The paper has very clear motivation and intuition for each method components. For example, using PointSAM backbone as initialization to adopt "2D SAM image prior" is a clever design and naturally fits. 

The authors also conducted extensive experiments to compare with baselines, and also ablations to show the validness of each proposed components.

The overall presentation of the paper is good and well-structured, with clear transition from motivation, to method design, to experiment part to show the validness of the proposed method.

### Weaknesses
I think the main weakness is that some details of the method / the system is not that clear in the current paper. I'd like authors to add more explanations on these parts:

1. What's the detailed model architecture to generate mask given point prompt, in the architecture of the network and also the shape format of all the input-output here.

2. What's the mechanism of incorporating previous point prompt? Both as a positive point prompt or a negative point prompt.

3. What's the detailed process of the IoU head in the network?

In general I think the network architecture and process can be elaborated more, to make it more clear to the readers.

### Questions
Please see the weaknesses above. Besides that, some other questions:

1. Will the non-clean meshes, or meshes with broken geometry in the dataset harm the training process? For example many assets in Objaverse are "not semantically meaningful" assets with e.g. a flat thin surface. Is there any process to avoid using these data, or these data are causing some negative effect.

2. It'd be good to show more qualitative results on "inner structure" of the model's output comparing to baselines, to showcase the advantage of trained with native 3D data.

3. It'd also be good to show some visualization results about the failure case.

In general I think this is a good paper, with aforementioned questions and weaknesses being answered/addressed, I'll raise my score accordingly.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper introduces PartSAM, a promptable 3D part segmentation model trained on native 3D data that outperforms state-of-the-art methods on both interactive and automatic part segmentation tasks. The work also contributes a large-scale 3D part dataset comprising 180K shapes created by merging existing part labels into semantically consistent parts, along with 320K shapes annotated using pseudo-labels.

### Strengths
1. PartSAM achieves significant quantitative improvements over existing methods on its benchmarks for both interactive and automatic part segmentation tasks.
2. In the data curation process, the merging criteria that exclude assets failing to form semantically meaningful parts is an excellent filtering strategy. Although automatic part segmentation is performed in a class-agnostic manner, ensuring that segmented parts can be assigned appropriate semantics remains valuable and well-motivated.

### Weaknesses
1. PartObjaverse-Tiny contains only 200 shapes across 8 categories, which is insufficient to convincingly demonstrate model performance. While evaluation on the PartNetE dataset (1,906 shapes) provides additional evidence, evaluating on larger and more diverse benchmarks such as 3DCOMPAT++ [1] would significantly strengthen the claims.
2. Although the ablation study in the Appendix demonstrates performance improvements brought by the model-in-the-loop annotation, the results are only evaluated on the small PartObjaverse-Tiny dataset. The paper would benefit from more comprehensive analysis demonstrating annotation quality. Given that the data curation process emphasizes semantically meaningful parts, I believe 3DCOMPAT++'s part definitions would be more appropriate for evaluation, as they possess stronger semantic grounding.

[1] 3DCOMPAT++: A comprehensive dataset for 3D object understanding with fine-grained part annotations

### Questions
1. Given the weakness 1 and 2, would it be possible to evaluate performance on the car and airplane categories (68 meshes in total) from 3DCOMPAT++ [1] and compare the results with PartField at this stage? It would be particularly valuable if the evaluation could be performed on both fine-grained and coarse part levels.
2. PartField demonstrates emergent cross-shape consistency when exploring feature space. Could you provide both qualitative and quantitative (if possible) results showing whether PartSAM exhibits similar cross-shape consistency, along with comparisons to PartField?
3. As mentioned in the model-in-the-loop annotation section, PartField suffers from limited controllability. In my experience, it sometimes cannot even segment all four wheels of a bus. Given this limitation, is using PartField outputs as candidate labels truly a reliable choice to obtain semantically meaningful annotations? Are there specific technical safeguards in place to prevent such failures from compromising pseudo-label quality? 
4. Previous 2D-lifting approaches often suffer from limitations such as 3D inconsistency in 2D models. The motivation for a native 3D solution should therefore be to avoid such error modes inherent in 2D models. Given this context, I am curious why it is necessary to retain 2D priors from SAM in the dual-branch encoder design?

[1] 3DCOMPAT++: A comprehensive dataset for 3D object understanding with fine-grained part annotations

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
3

### Summary
This paper introduces PartSAM, a 3D part segmentation model that avoids lifting 2D masks, supports interactive prompts, and offers a “segment-every-part” mode for full-shape decomposition. Architecturally, it combines a triplane dual-branch encoder (one frozen to preserve SAM-derived 2D priors, one learnable for native 3D) with a prompt-guided decoder, enabling controllable, geometry-aware segmentation including interior structures. A model-in-the-loop pipeline scales supervision to >5M shape–part pairs from large 3D asset repositories, yielding diverse, fine-grained labels. Experiments show decisive SOTA gains over Point-SAM/PartField on PartObjaverse-Tiny and PartNet-E.

### Strengths
I like the paper's architecture–data efforts (SAM-like decoder + dual-branch triplane encoder + large 3D supervision pipeline) yields generalizable segmentation that extends to interior parts. Although no novel technical contributions, but I do like the whole pipeline and believe such model-data efforts are good to community if open-sourcing.

The evaluation is extensive, including interactive/automatic settings, ablations, complexity analysis. The results show consistent over strong baselines, including SAMPart3D and PartField.

### Weaknesses
The part-level are ambitious or the part definition is unclear. For example, the left-second human-horse case in Figure 1 segment the tail into two weird chunks, while the hair and the horse head are not separate. Also, the current multi-scale is generated by varying NMS
threshold value, and also the beginning is by using PartField along with different clustering number. I feel this is a fundamental limitations for the current setup.

The performance of SAMesh is actually quite good as well, where the proposed method is superior compared to SAMPart3D and PartField, as the boundaries is more clear. However, SAMesh also has clear boundaries. The proposed method is more fine-grained in large pieces, while SAMesh tend to segment into large pieces, where I feel can tune the 2D segmentation model fine-grained level to adjust this point. Also, the grenade in SAMesh is very fine-grained. Therefore, I do not believe the proposed method is absolutely better than SAMesh. For this point, I actually want to see more qualitative comparisons between the proposed method and SAMesh with randomly sampling. Not cherry-pick, but randomly sampling can give reader better understanding of the performance differences. Also, there is no need for the proposed method absolutely better than SAMesh as two methods are driven from very different technical backbones. I just encourage the author to conduct the comparisons more systematics, and show the limitations for both the SAMesh and the proposed method. This will not harm the paper at all.

### Questions
Plz address the concerns raised in the weakness and ethics concerns sections.

### Soundness
3

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
This paper proposes a novel method, PartSAM, for 3D part segmentation. The high-level idea of PartSAM follows that of 2D SAM or PointSAM — it trains a feedforward model that takes a point prompt as input and outputs point masks. Unlike recent 3D part segmentation methods that leverage supervision from 2D SAM, this model is among the first to utilize 3D-native part information embedded in the raw mesh.

### Strengths
1. The paper addresses an important task, and the proposed method is well-motivated.

2. The idea of leveraging scene graphs and “model-in-the-loop annotation” is clever.

3. The paper is generally well written and easy to follow.

4. The proposed method outperforms previous methods.

5. The authors provide insightful ablation studies in the supplementary materials.

### Weaknesses
1. **Method novelty and comparison with existing work.**
   From a methodological standpoint, the high-level idea of the proposed model and training scheme closely follows that of 2D SAM, and therefore lacks substantial novelty or conceptual breakthrough. Moreover, similar attempts to extend 2D SAM to 3D already exist (e.g., PointSAM). Can the authors train PointSAM on the newly collected 3D dataset? Would it yield better or worse results? What are the specific advantages of the proposed network architecture or training scheme compared to PointSAM or 2D SAM?

2. **Contribution of large-scale 3D native data.**
   I understand that one of the main contributions of this paper lies in training on a large-scale 3D-native dataset. However, I would like to see a deeper quantitative and qualitative analysis of the data quality and its impact on model performance. For example, what are the concrete benefits of 3D-native data compared to 2D-lifted data? Is it more accurate, or does it offer other advantages such as better generalization or consistency? It would also be helpful if the authors could provide more discussion on whether the current data pipeline can reliably produce high-quality part annotations, and what remaining challenges exist in the data curation process.

3. **Experiments on data scale and quality.**
   I recommend adding further experiments to investigate the influence of data quantity and quality, such as scaling curves showing performance variations with different training data sizes.

4. **Impact of connectivity information.**
   I understand that connectivity information may play an important role in part segmentation. The current experiments only report results without considering connectivity information and appear to adapt the baselines accordingly. For a more comprehensive comparison, it would be beneficial to include both setups—*without* connectivity and *with* connectivity. In many scenarios, it is desirable for the model to utilize connectivity cues to produce more accurate and coherent part segmentations. Such comparisons would provide valuable insights into the model’s capability and potential advantages under different conditions.

### Questions
1. **Data release.**
   Will you release the data processing scripts and the processed datasets with part annotations to facilitate future research and ensure reproducibility?

2. **Use of connectivity information.**
   I noticed that the paper mentions the proposed method can leverage connectivity information “by applying graph cuts to the segmentation results for one iteration.” However, this description is still unclear. Could you please provide more details on how graph cuts are integrated into the pipeline?

3. **Hierarchical part decomposition.**
   Could you elaborate on how hierarchical part decomposition is achieved? Specifically, how are finer-grained parts derived from coarser-level segmentations, and what criteria or supervision signals are used to determine the part hierarchy?

4. **Runtime and efficiency.**
   Regarding the reported timing results, does the measurement correspond to the inference time for a single-point prompt, or to the runtime of the full automatic segmentation process that samples multiple point prompts? How many point prompts are typically used for automatic segmentation, and how does this number influence the overall runtime and efficiency?

### Soundness
3

### Presentation
3

### Contribution
3

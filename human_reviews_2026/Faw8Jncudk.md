# C3R: Channel Conditioned Cell Representations for unified evaluation in microscopy imaging

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 4, 4, 4

## Abstract
Immunohistochemical (IHC) images reveal detailed information about structures and functions at the subcellular level. However, unlike RGB images, IHC datasets pose challenges for deep learning models due to their inconsistencies in channel count and configuration, stemming from varying staining protocols across laboratories and studies. Although existing approaches build channel-adaptive models, they do not perform zero-shot evaluation across IHC datasets with unseen channel configurations. To address this, we first introduce a structured view of cellular image channels by grouping them into either context or concept, where we treat the context channels as a reference to the concept channels in the image. We leverage this view to propose Channel Conditioned Cell Representations (C3R), a framework that learns representations that transfers well to both in-distribution (ID) and out-of-distribution (OOD) datasets which contain same and different channel configurations, respectively.  C3R is a two-fold framework comprising a channel-adaptive encoder architecture and a masked knowledge distillation training strategy, both built around the context-concept principle. We find that C3R outperforms existing benchmarks on both ID and OOD tasks, while yielding state-of-the-art results on CHAMMI-ZS; a zero-shot-style adaptation of the CHAMMI benchmark. Our method opens a new pathway for cross-dataset generalization between IHC datasets, with no need for retraining on unseen channel configurations.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper presents an architecture and a training framework to create microscopy image models that can be reused across different imaging configurations with different channels. The main idea of the architecture is to group channels in two sets: context channels and concept channels, which is biologically motivated by the way some imaging experiments are conducted. Context channels serve as a reference frame to observe variations in the information collected in concept channels.

The proposed method was trained on the HPA dataset and evaluated in out of distribution tasks. Out of distribution in this paper is focused on channels observed during training, which is a rather narrow definition. The presented experimental results indicate promising performance of the proposed approach, including evaluations in the JUMP dataset, and in the CHAMMI benchmark.

### Strengths
* The idea of grouping channels in context and concept is interesting, and biologically motivated.
* Based on this idea, the paper proposes an inductive bias to the architecture of ViTs and model training.
* The paper makes a good presentation of previous work.
* The experiments use established datasets to evaluate performance and investigate the properties of the models.

### Weaknesses
The paper has several issues and the experimental results do not support the conclusions and claims of this paper.

###  Context / Concept

* The main technical limitation of the proposed approach is that the definition of context and concept channels needs to be manually defined. The decision may be very arbitrary in practice.
* The quantification of context and concept seems to match the hypothesis, but it is not 100% supported by the data (as discussed in the Supplementary material). The separation can be artificial, and even if it’s quantitative, the paper is not presenting a solid method to automatically identify it in practice.

### Training
* The models and experimental evaluation are based on training with a single dataset (line 328). 
* With a single training dataset, it is unclear how the model learns generalizable representations to other channel configurations. Specifically, the HPA dataset seems to have only one concept channel during training. 
* The formulation of the model follows standard channel-wise token processing, originally proposed in the Channel-ViT paper and followed by others (Eq. 4 to 9). This architecture can be trained with varied numbers of channels. It is unclear why the proposed model is only trained with the fixed channels in HPA. 
* It is not completely clear if the training algorithm is SSL or weakly supervised. There is no clear definition of the objectives and losses used in their models and the baselines. 
* The focus on out-of-domain generalization is narrow to different channels only, when there are many other aspects of OOD generalization that can be studied. This includes different cell lines, treatments, batches and so on. The experimental designed is not clean in testing these types of OOD evaluations because of the restriction of training only on one dataset.

### Baselines

* Missing baseline evaluations. The Base-CP and Base-SC models are retrained with the same HPA data but with different channel settings to match the target dataset. Evaluation of models tailored to the channels and with the data of the specific task is important (specialized models).
* Missing baselines. Other pre-trained models such as OpenPhenom or DINOv2 were not included in the evaluation. The former is channel adaptive, and the other can be adapted as presented in Figure 2 (line 182). It is unclear why these models were not used.
* Baseline models seem suboptimal or impaired. ChannelViT and DiChaViT were trained with two channels at a time (line 1029). The models have the capacity to train with variable channel lengths, but the study is limiting this by arbitrarily choosing only 2, while the proposed models get to see 4 channels.
* DINO4Cells and SubCell were said to be the pre-trained available models (lines 350 and 351), but Table 2 indicates they are re-trained to match the JUMP-CP channel configuration. It is unclear how to interpret these results.
* Base-SC is trained with antibody loss, which is a supervised or weakly supervised loss (lines 328-332). It does not make sense to use a supervised loss with a single channel.  
* The results in Table 3: the models are supposed to be trained with a 4 channel dataset, but the baselines only get to see 3 channels (line 331). In addition, the target dataset is 5 channels, so what channels are used and what channels are dropped is unclear, and it is an unfair comparison.

### CHAMMI benchmark

* CHAMMI is already a zero-shot benchmark, which uses K-NN evaluation for all 9 tasks (IID and OOD), and can be evaluated even with ImageNet pretrained models without any training. The statement that CHAMMI needs zero-shot style adaptation is incorrect, especially under the presented experimental setup (training only with HPA).
* The introduction of CHAMMI-ZS (zero shot) is highly misleading, because a specialized, two-layer MLP is actually being trained (lines 341-344). Zero-shot evaluation means no training, resulting in a conceptual and experimental mistake. 
* The correct way of using CHAMMI for zero-shot evaluation is to compute features in all images and run the kNN evaluations. Those are the results that should be reported, specially if the paper claims generalization to unseen channels.


### Incorrect terminology
* Immunohistochemistry Channels => only one of the channels in HPA is IHC, the other channels are fluorescent channels. None of the channels in JUMP are IHC. This is the wrong terminology to refer to the datasets used in the presented experiments.
* Models do not have channels. The input images have channels, the models are functions that only have parameters. Statements such as “encourage the channels” (lines 093, 295) do not make sense, because the channels are fixed data points rather than parameters.
* In an experiment without re-training or fine-tuning, what gets frozen are the parameters of the model not the features (lines 334-340, 344, 424, etc). 

### Other issues
* Figure 4 should keep together the bars of original and flip, rather than mAP and kNN. The quantities that need to be compared are split apart in the current layout. Also, the scale is misleading, giving the impression that the difference is large when it is actually very small.
* In Table 4, why is the Base-HPA worse than h_c alone in the HPA tasks? Is the baseline suboptimal? Base-HPA is supposed to get all the information needed to solve the HPA tasks in almost the same way as h_c. Is there any explanation for this difference?
* Figure 3 displays 5 channels (3 blue and 2 red) but in practice the experiments are all with 4 channels.
* Typo in line 203: $p_k^c$ when it should be $p_c^k$.
* Sloppy notation. In equation 8, the x parameters are missing the i and j super index respectively (e.g., $\hat{x}^i_{c1}$).

### Questions
* If the concept/ context idea is robust enough, why there is not a method to automatically detect the type?
* Why are all the experiments trained only on HPA? Does the method fail if trained on other datasets? This severely limits the potential of this study and the correctness of many reported experiments.
* Why is the method not trained with varied numbers of channels but rather with 4-fixed channels from HPA?
* Is there a limitation that prevents the model to be trained with disparate number of channels? In theory not, but why was this not tested?
* What is an antibody loss and what are the definitions of the training objectives? This is not clearly defined, and it is unclear whether the models are self-supervised or supervised.
* What does it mean to re-train baseline models DINO4Cells and SubCell to match the JUMP-CP channel configuration? When is this necessary?
* When dropping channels from the model (e.g., 3-channels for JUMP-CP), which ones are dropped and which ones are used and why?
* When adapting the proposed model trained with HPA, how are the assignments of context / concept channels made? Was this quantitative in CHAMMI, for instance? Where is this reported and why these choices?
* In the CHAMMI evaluation, is an MLP trained for each subset of the benchmark? The proposed model can be channel-adaptive, but an MLP is not. The details of how this is done are unclear.
* Why not conduct real zero-shot in CHAMMI? Why there is a need to train a two-layer MLP if the features are obtained with a frozen model? Why not just run the k-NN evaluation?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces  C3R, a novel framework for representation learning fluorescence microscopy images. The authors identify an inherent context–concept structure in microscopy channels, where some channels (e.g., nucleus, ER) provide structural context while others convey experiment-specific information (typically about the protein under study). Building on this insight, C3R combines (i) a Context–Concept Encoder (CCE) that processes channel groups separately before joint integration, and (ii) a Masked Context Distillation (MCD) pretraining scheme that promotes robustness to missing or unseen channels. The method enables zero-shot generalization across datasets with novel channel configurations. Experiments on HPA, JUMP-CP, and CHAMMI-ZS benchmarks show that C3R outperforms prior methods (DINO4Cells, SubCell, ChannelViT, DiChaViT) both in in-distribution (ID) and out-of-distribution (OOD) settings.

### Strengths
1. Important and relevant problem of fluorescence microscopy representation learning 
2. Overall well-written and easy to read. 
3. Conceptually original idea (context/concept channel separation) 
4. Comprehensive ablation study

### Weaknesses
1. Conceptually, it is not so clear how to distinguish between context and concept channels. First, this distinction depends on the study design: the same channel might provide context for one assay and content for another. For instance, nuclei are context information for subcellular localization screens, while they provide the relevant readout when studying cell cycle, nucleoli, aneuploidy or cell death.  
2. Related to this, I am not entirely convinced by the analysis and results provided in section 3.1. First, the authors make no distinction between intra-image heterogeneity (if for instance localization patterns are less marked or feature higher inter-cell variability) and inter-image heterogeneity (if perturbations affect the localization more). Second, the authors show only summary metrics for $P_c$ and $H_c$, thereby occluding differences between context channels. Indeed, the nucleus channel might heavily influence both metrics, while the ER could be more similar to a content channel. Finally, whether a channel provides context or content depends on which biological process I am interested in (see my point 1). This is a problem, especially since, as the authors have demonstrated, switching channel categories further degrades performance. 
3. The SSL MCD contribution is relatively incremental, and the validation (Table 5) sometimes depends on specific conditions, limiting robustness. 
4. The work does not leverage recent state-of-the-art SSL methods such as DINOv2/3 or MocoV3, relying instead on iBoT, which may restrict comparative performance. 
5. The approach lacks scaling to be considered truly generalizable; it appears more as a proof of concept than a fully deployable solution. 
6. There appears to be some confusion between Immunohistochemistry (IHC) and Immunofluorescence (IF). 
7. I would have expected a short explanation of the downstream tasks and biological objectives. Indeed, the datasets and associated tasks might not be known to the ICLR community.

### Questions
1. The authors use the term Immunohistochemical (IHC) throughout the text, which can be easily confounded with a modality used in digital pathology, usually not based on fluorescence. It would be better to use an unambiguous term, such as Immuno-fluorescence (IF) throughout the text. 
2. Could the authors propose a systematic rule for assigning channels between context and concept? 
3. Did the authors try SOTA SSL methods (e.g., DINOv2/3, MocoV3)? Is there a rationale for choosing iBoT? 
4. In Figure 2, could the authors add a legend to facilitate the distinction between Concept and Context channels? Additionally, it is not very clear that Concept channels exhibit higher variance than Context channels from the UMAPs.  
5. In Figure 4, it is noted that flipping the channels during inference degrades the metrics. Did the authors also train a model with flipped channels to assess the impact on performance? 
6. Figure 6: Can the authors explain how the error bars are calculated, and why are they only displayed for the HPA-Loc dataset? Moreover, the  depth evaluation was performed by fixing the number of layers at 4. Can the authors justify this choice? Figure 6 seems to indicate that the optimal depth is 2. 
7. Figure 6 seems to indicate that, overall, larger networks perform better. Have the authors tested more complex architectures to further improve performance?

### Soundness
3

### Presentation
3

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
This paper addresses the challenge of applying deep learning to microscopy (IHC) images, which have inconsistent channel counts and configurations across datasets. This inconsistency normally prevents a model trained on one dataset from working on another without retraining.

The authors introduce the "context-concept principle," dividing channels into:
* Context: Stable structural references (e.g., Nucleus).

* Concept: Variable, experiment-specific information (e.g., Protein).

Based on this, they propose C3R, a framework with two parts:

* Context-Concept Encoder (CCE): A new architecture that processes these channel groups separately before merging them, enabling generalization to unseen channel configurations in a zero-shot manner.

* Masked Context Distillation (MCD): A training strategy that forces the model to learn from a limited set of context channels, improving representation robustness.

### Strengths
* The paper addresses a real issue in the field of image representation learning for microscopy images: modern image representation learning architectures are designed for natural RGB images, which always come in the three-channel format, where there is no significantly distinct information across channels.

* The paper introduces a new and intuitive way to structure microscopy data by grouping channels into "context" and "concept". The model's success should speak to the power of grouping channels accordingly.

* Using community-accepted benchmarks to evaluate results against other models

### Weaknesses
* The paper claims that its framework is the first of its kind to demonstrate strong zero-shot evaluation for this problem, but I found the following paper that purports to do the same, as is also similarly channel adaptive [1].

* The model's ability to evaluate a new OOD dataset relies on the assumption that the new dataset's channels can also be separated into "context" and "concept" groups. While the authors note this is true for the most common public IHC datasets (HPA, JUMP-CP, WTC-11, etc.), they have not explored datasets that may not follow this principle.


[1] L. Phillips and R. Donovan-Maiye, "CellRep: A multichannel image representation learning model," in Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, Jun. 2025, pp. 4312–4318.

### Questions
* Have you explored methods to learn this context-concept grouping automatically?

* MCD's success implies context channels are redundant, yet this robustness fails on the OOD JUMP-CP dataset. Does this mean the rules of context redundancy are not generalizable, and what does that imply for the 'context-concept principle' on new, unseen datasets?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
In this paper, the authors introduce a structured approach to training a channel adaptive model that can generalize well in zero-shot settings that are both in-distribution (having the same channels) and OOD (different channels during training and inference). They first introduce the Context-Concept principle by segregating channels into context channels (structural channels) vs concept channels (non-reference information). They propose Channel Conditioned Cell Representation (C3R) that incorporates the structural separation during training that generalizes well to ID and OOD at inference time. They also show their model performs state of the art on CHAMMI-zero-shot adapted dataset.

### Strengths
* Improved training strategy: Masked Context Distillation with concept context segregation is intuitive and adds biological data structure prior to training to an extent. 
* Comprehensive evaluation and improved performance on several benchmark datasets (HPA, JUMP-CP, CHAMMI-ZS) in both in distribution and OOD settings.

### Weaknesses
* The assignment of concept or context groups are rather arbitrary or manually labeled subjectively. 
* Random channel masking across Student/Teacher (i.e. channel masking as an augmentation baseline) is missing and the contribution of MCD (masking context channels specifically) is unclear. 
* The contribution is very specific to IHC/fluorescence microscopy images and the subjective nature of context/concept selection makes it not generalizable and scalable to operate as a methodology.
 * CHAMMI-ZS benchmark doesn't seem to be truly zero-shot as there is a dataset specific MLP heads used for evaluating on the dataset.

### Questions
* What was the procedure for the segregation of context vs concept channels? It feels rather arbitrary than a specific methodology. The per group parity and entropy metrics do not necessarily capture the contextual or conceptual nature of the channels.
* Have you compared results on Masked Context Distillation with masking just concept channels or allowing masking all channels between student and teacher (Table 5 doesn't seem to answer that question clearly)? How much does Masking just the context contribute to performance compared to masking as an augmentation?

### Soundness
3

### Presentation
3

### Contribution
2

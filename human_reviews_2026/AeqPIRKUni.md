# Animal behavioral analysis and neural encoding with transformer-based self-supervised pretraining

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
The brain can only be fully understood through the lens of the behavior it generates--a guiding principle in modern neuroscience research that nevertheless presents significant technical challenges. Many studies capture behavior with cameras, but video analysis approaches typically rely on specialized models requiring extensive labeled data. We address this limitation with BEAST (BEhavioral Analysis via Self-supervised pretraining of Transformers),  a novel and scalable framework that pretrains experiment-specific vision transformers for diverse neuro-behavior analyses. BEAST combines masked autoencoding with temporal contrastive learning to effectively leverage unlabeled video data. Through comprehensive evaluation across multiple species, we demonstrate improved performance in three critical neuro-behavioral tasks: extracting behavioral features that correlate with neural activity, and pose estimation and action segmentation in both the single- and multi-animal settings. Our method establishes a powerful and versatile backbone model that accelerates behavioral analysis in scenarios where labeled data remains scarce.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper suggests to use InfoNCE contrastive loss to learn better representations from the video, where similar and different embeddings are defined based on the time proximity.
The model in pre-trained jointly with InfoNCE and reconstruction (demasking)  loss (MSE) and later fine-tuned for a specific task, such as a)neural encoding; b) pose estimation; c)action segmentation.

### Strengths
1. The paper is very well-written and nice to read
2. The paper brings recent advances in self-supervised learning toward animal tracking community and tests the proposed method on various datasets from multiple labs.
3. On some tasks like Action segmentation (table 3) the method shows a clear improvement 
4. The authors consistently provides error bars (SEM or std) in all plots and tables
5. Realistic evaluation: The limited labeling settings (100 labeled frames as described in page 7 lines 362-363) is close to real life evaluation scenarios.

### Weaknesses
1. A lot of references are missing, I would highly inspire the authors to do another iteration on the related works
The paper basically introduced contrastive learning on spatiotemporal video chunck, which have already been done in [1] also using InfoNCE. Additionally, Triplet loss [2] another conceptually close (to attract and repel to the anchor point) and is worth mentioning and maybe benchmarking.
On the animal tracking / pose estimation side, SLEAP [3], DeepPoseKit [4], TRex [5] are missing.
Corresponding baselines (or justification if they are not needed) is missing as well.

2. Its not clear why contrastive temporal loss even works for highly repetitive behaviour - e.g. for the dataset of a mouse walking paws - this is supposed to be very close to each other no matter is the points are close in time or not, as far as the paws are arranges same ("left or right paw step"). A bit of this discussion together with the frame selection process is is presented in the Appendix B2 but I do not find it clear enough and Table 4 and 5 often do not show statistically significant improvement (eg the improvement is 3rd digit after the dot, while the std is variable in the 1st digit after the dot, as in Table 4 1st column).

3. The improvements in tables seems to be borderline statistically significant (within std error bars, e.g. table 1 reports the improvement in the 2nd digit behind the dot while $\pm \text{std} \approx 0.1$, same comments for tables 4 and 5 was already done above).


References:  
[1] Qian, Rui, et al. "Spatiotemporal contrastive video representation learning." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2021.  
[2] Schroff, Florian, Dmitry Kalenichenko, and James Philbin. "Facenet: A unified embedding for face recognition and clustering." Proceedings of the IEEE conference on computer vision and pattern recognition. 2015.  
[3] Pereira, Talmo D., et al. "SLEAP: A deep learning system for multi-animal pose tracking." Nature methods 19.4 (2022): 486-495.  
[4] Graving, J. M. et al. DeepPoseKit, a software toolkit for fast and robust animal pose estimation using deep learning. eLife 8, e47994 (2019).  
[5] Walter, T. & Couzin, I. D. TRex, a fast multi-animal tracking system with markerless identification, and 2D estimation of posture and visual fields. eLife 10, e64000 (2021).

### Questions
1. *(most crucial)* Could you provide some qualitative experiments showing that the temporal contrastive loss actually selects different enough negative pairs? Have you explored any alternative sampling strategies for the frames ?
2. In lines 482-484 you write *"The black-box nature of VIT embeddings presents interpretability challenges in scientific contexts where transparent representations like pose estimates are often preferred."*. Could you please clarify how exactly is your model more interpretable?
3. How do you match frequencies between behaviour and neuronal recordings for Fig2? Spiking neuronal data should be much higher frequency than behavioural timeseries and calcium data should be slower, especially for Facemap dataset, where behavioural videos should have been 50Hz to catch whiskers motions.
4. What is the intuition/ interpretation behind Bits Per Spike metric, especially in the calcium imaging settings?
5. Why SLEAP and DeepLabCut are not used as baselines for pose estimations? (DeepLabCut might be seen as pretrained ResNet version but having simple UNet comparison is missing, which both of them do, is missing). If you have tried it and it did not work on 100 frames, I think it should be mentioned in the appendix given how common these tools are in the community.
6. How would you expect you method to work for animal tracking for in the wild (e.g. monkeys tracking in the jungles), where the motion is much less periodic and more various but the background could be stationary as well (e.g. wind would move the leaves)? 

Minor:
1. You report training *"taking approximately 25 hours on 8 Nvidia A40 GPUs"* in line 202. This is without ImageNet pretraining and simultaneously on all datasets (e.g. mice and fish together)? Why does it take so long if you start from the weights after ImageNet pretraining? Do you have a warmup stage before cosine annealing?
2. Why do you train CEBRA models not jointly across sessions ? (as mentioned in App. C)
3. For Fig.4 - could you show non-normalized confusion matrixes (panel B) because it looks like for the middle just the majority class gets switched? Could you also please provide the weighted accuracy ?
4. For Fig.4 panel A - what is ensembled for PCA? should not it be deterministic if only seeds are varied? If yes - how is there a difference for "CalMS21" dataset?

### Soundness
3

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
4

### Summary
This paper introduces BEAST, a self-supervised learning framework that combines masked autoencoding (MAE) and temporal contrastive learning to pretrain vision transformers on animal behavior videos. The authors demonstrate the utility of BEAST across three key neuro-behavioral tasks: neural encoding, pose estimation, and action segmentation. The method is evaluated on multiple datasets and species, showing consistent improvements over existing baselines, especially in settings with limited labeled data.

### Strengths
Comprehensive Evaluation: The paper thoroughly evaluates BEAST across multiple tasks (neural encoding, pose estimation, action segmentation) and datasets (mice, fish), demonstrating its versatility and robustness.

Neural Encoding Focus: The inclusion of neural encoding as a downstream task is a significant and compelling contribution, linking behavioral video analysis directly to neural activity—a fundamental goal in neuroscience.

Efficient Use of Unlabeled Data: BEAST effectively leverages unlabeled video data through self-supervised pretraining, reducing the need for extensive manual annotation.

Ablation Studies: The paper includes extensive ablations (e.g., frame selection, pooling strategies, loss weighting) that provide valuable insights into the design choices.

### Weaknesses
1. Limited Discussion of Related Self-Supervised Methods: While the paper discusses general self-supervised learning (SSL) methods like MAE and contrastive learning, it does not adequately address existing SSL approaches specifically designed for animal behavior. For example:

    ConstrastivePose: A contrastive learning approach for self-supervised feature engineering for pose estimation and behavorial   classification of interacting animals.
    Tianxun Zhou, Calvin Chee Hoe Cheah, Eunice Wei Mun Chin, Jie Chen, Hui Jia Farm, Eyleen Lay Keow Goh, Keng Hwee Chiam.
    bioRxiv 2022.11.09.515746; doi: https://doi.org/10.1101/2022.11.09.515746

    Selfee: Self-supervised Features Extraction of animal behaviors.
    Yinjun Jia, Shuaishuai Li, Xuan Guo, Junqiang Hu, Xiao-Hong Xu, Wei Zhang.
    bioRxiv 2021.12.24.474120; doi: https://doi.org/10.1101/2021.12.24.474120

    Domain-Adaptive Pretraining Improves Primate Behavior Recognition.
    Felix B. Mueller, Timo Lueddecke, Richard Vogg, Alexander S. Ecker.
    arXiv:2509.12193; doi: https://doi.org/10.48550/arXiv.2509.12193

    A more thorough discussion of these works would better contextualize BEAST’s contributions.

2. Performance on IBL-Whisker Dataset: The results show that BEAST does not outperform DINOv2 on the IBL-whisker dataset (Table 4). The authors should discuss why this might be the case (e.g., DINOv2’s stronger pretraining or architectural advantages?). Also, whether further pretraining DINOv2 on IBL data would close or widen the gap.

3. Missing DINOv2 Baselines for some tasks: While DINO is included in pose estimation and DINO v2 is evaluated in neural encoding, DINOv2 is not evaluated for behavior classification or pose estimation. It is confusing. Given its strong performance in other domains, it should be included as a baseline for a fair comparison.

4. Technical Novelty: The core components of BEAST (MAE + contrastive learning) are based on ViC-MAE, which is not novel. The main contributions lie in the application-specific adaptations (e.g., frame sampling, domain-specific pretraining) rather than the underlying architecture.

### Questions
1. Why does BEAST largely underperform compared to DINOv2 on the IBL-whisker dataset (in terms of RRR)? Would pretraining DINOv2 on IBL data improve its performance further?

2. Have you considered using only patches containing the animal (via segmentation or attention) for behavior classification? This may help reduce background noise.

3. Why was DINOv2 not included as a baseline for behavior classification and pose estimation? Its strong performance in other domains warrants its inclusion.

4. How does BEAST compare to DINOv3, which was released after this submission?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper introduces BEAST (BEhavioral Analysis via Self-supervised pretraining of Transformers), a domain-tailored framework that pretrains experiment-specific ViTs on unlabeled animal-behavior videos by combining masked autoencoding with a temporal contrastive objective. A key ingredient is a frame selection/sampling strategy designed for static backgrounds with movement-driven variation. The pretrained backbone is evaluated on three tasks across multiple datasets/species. BEAST delivers competitive or superior results to strong baselines while reducing reliance on large labeled sets or keypoint-first pipelines.

### Strengths
1. Cleverly tailors self-supervised pretraining to behavioral videos (static background, movement-driven variation) via selection/sampling strategies. 
3. Evaluates frozen-backbone features on segmentation and zero-/few-shot neural encoding, with systematic ablations.
3. Clear writing.
4. On action segmentation, BEAST bypasses pose-estimation training while matching or exceeding keypoint-based systems, reducing months-long annotation pipelines.

### Weaknesses
1. Cross-domain generalization. The approach emphasizes single-paradigm pretraining; comprehensive tests across cameras/environments/species would strengthen claims of generality. 
2. Efficiency. The paper positions BEAST as more efficient than native video models but does not report controlled FLOPs/memory/runtime.
3. Interpretability. Beyond showing CLS superiority for pretraining, adding feature visualizations or sensitivity analyses linking learned features to anatomical/behavioral elements would aid scientific adoption.

### Questions
see weakness

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
The authors propose BEAST, a transformer model for behavioral analysis with neuroscientific applications. The model uses a combination of a masked and contrastive objective on raw video frames, and can be used for three downstream tasks (neural encoding, action segmentation, pose estimation). The authors claim benefits in neural encoding over baseline methods for video/behavior encoding, improved zero shot neural encoding performance over a ViT baseline, improvements in keypoint estimation over a ViT baseline, and in action segmentation vs. the SimBA model. The supplementary material contains a number of additional experiments, variations and ablations of the proposed approach.

### Strengths
- Ablations and comparisions are very extensive and clearly outlined
- Paper is very well written. Dense, but very informative and to the point.
- The figures have high quality and are far above the average of ICLR papers
- The model has convincing performance across the different tasks.

### Weaknesses
While the method presented seems powerful for a variety of tasks, the evaluation as-is is currently too weak; it seems like BEAST is largely building on existing video-pre-training schemes. This in itself might be fine, but from e.g. Table 1 and 4 it seems that even frozen backbone models are suitable for solving the neural encoding tasks.

The authors need to better delineate what their methodological contribution is, and what it adds over a strong baseline model. I would e.g. consider doing a bit of finetuning with DINOv2, or VideoMAE, or TCN, on the authors' dataset, and check if this might surpass the performance of BEAST on one of multiple of the tasks; the tables suggest that this is likely. The paper might still be a nice contributions, but should be rephrased slightly; right now it appears like the exact combination of training objective is causing a performance gains, which I am critical about.

Some weaknesses I see are:

1. Figure 2: Choice of baselines is questionable. Motion energy and and PCA are good as "naive" baselines, but the CEBRA result feels obvious given that the backbone for BEAST is a ViT model, and for CEBRA the authors used a small convnet. The better comparison would be to use the same backbone, and only vary the training objective. The authors should also pull the baseline model (VIT-M) in Table 1 into this figure/barplot for better contextualization what their particular training objective contributes to this task.
2. Figure 3: Same concerns regarding baselines. BEAST is compared against a ResNet-50 model. What about other pose estimation algorithms that are widely used, e.g. DeepLabCut, which is also referenced in the text? These would be relevant comparisons for practitioners.
3. More baselines for action segmentation could be discussed and benchmarked; right now the result seems anecdotal.
4. While the results in the figures are nice, it seems like the authors created these benchmark situations and evaluated; it would be good to discuss how the authors validated their baselines, and if there are any previously published benchmarks BEAST can be applied on (and please excuse if there is a supplementary result that I overlooked already doing this)
5. The authors cite "temporal contrastive learning (Chen et al., 2020)", which is not the right reference for temporal contrastive learning. TCL and PCL might be better algorithms to cite, or even CPC (2018).

### Questions
1. I might have missed this in the numerous appendix results, but is there an ablation on the role of the two training objectives (masking vs. contrastive learning)? Are both required for getting high performance?
2. The method seems incremental in terms of the pre-training objective, which is also transparently communicated in the appendix. Could the authors elaborate again what the most conceptually close prior algorithm to BEAST is, and clearly delineate what the changes are they added, and what the effect of this change is? This could also be made a bit clearer in the main paper.
3. Judging from the std/confidence intervals in most tables, the difference between ViT variants and BEAST is slim. Could the authors run stats on the different tables and communicate transparently which advances are actually signficant?
4. Is the keypoint benchmark, especially for the baseline model, in Figure 3 grounded in the literature? Same question for Figure 4.
5. For neural encoding, Table 4 suggests that frozen backbones perform almost on par with BEAST. What happens if e.g. VideoMAE would be finetuned on the same dataset? I find it likely that this might already surpass BEAST performance. Was such an experiment (using an existing state of the art pre-training paradigm on the same data) done?
6. There is no mentioning that the code (training and/or inference) and/or the model weights are planned to be open sourced. Is this planned?

Happy to discuss further.

### Soundness
4

### Presentation
4

### Contribution
2

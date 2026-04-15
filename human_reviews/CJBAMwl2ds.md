# Reasoning-Enhanced Object-Centric Learning for Videos

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 3

## Abstract
Object-centric learning aims to break down complex visual scenes into more manageable object representations, enhancing the understanding and reasoning abilities of machine learning systems toward the physical world. Recently, slot-based video models have demonstrated remarkable proficiency in segmenting and tracking objects. Although most modules in these models are well-designed, they overlook the importance of the effective reasoning module. In the real world, especially in complex scenes, reasoning and predictive abilities play a crucial role in human perception and object tracking; in particular, these abilities are closely related to human intuitive physics. Inspired by this, we designed a novel reasoning module called the Slot-based Time-Space Transformer with Memory buffer (STATM) to enhance the model's perception ability in complex scenes. The memory buffer primarily serves as storage for slot information from upstream modules, akin to human memory or field of view. The Slot-based Time-Space Transformer makes predictions through slot-based spatiotemporal attention computations and fusion. We demonstrated that the improved deep learning model exhibits certain degree of rationality imitating human behavior. This has crucial implications for understanding the relationship between deep learning and human cognition, especially in the context of intuitive physics.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper proposes a memory-enhanced Predictor in slot-based video object-centric learning (OCL) frameworks. The proposed Predictor, STATM, runs self-attention over slots at the same timestep and across timesteps. Experimental results on MOVi datasets show that STATM serves as a plug-and-play module to consistently improve the segmentation results of SAVi and SAVi++.

### Strengths
- The Predictor is indeed an under-explored component in SAVi-like models. This paper serves a preliminary attempt in this direction
- The ablation studies are thorough

### Weaknesses
My biggest concern is regarding the experimental settings:
- The reported performances of baselines are not with their best training configs. The reason "computation constraint" is not acceptable, as I am not sure with longer training & larger batch size, will the performance gain disappear
- As a result, the performance of STATM is not SOTA. SAVi++ reports a mIoU of 47.1 on MOVi-E, which is much higher than this paper
- Only comparing with SAVi and SAVi++ is also not enough. There have been several follow-up works since then, such as STEVE [1]. STEVE does not use the initial frame hint, but still produces meaningful segmentation results on MOVi datasets. I believe with initial frame hints, STEVE will even outperform SAVi++. So this baseline should be considered as well
- The experiments in the paper mainly focus on object segmentation. While it is an important outcome of OCL, the quality of learned object slots is another important aspect. I would suggest the authors to at least perform an object property prediction experiment on MOVi, e.g., follow the protocol in LSD [1]

[1] Singh, Gautam, Yi-Fu Wu, and Sungjin Ahn. "Simple unsupervised object-centric learning for complex and naturalistic videos." NeurIPS, 2022.

[2] Jiang, Jindong, et al. "Object-centric slot diffusion." arXiv preprint arXiv:2303.10834 (2023).

### Questions
Besides the questions raised in Weaknesses. I have a few minor questions:
- What is the computation and memory overhead STATM brings compared to SAVi and SAVi++? Please report the training memory and speed with and without STATM
- I agree that the current MLP or Transformer Predictor in SAVi/SAVi++ does not consider long-term temporal information. However, will adding an LSTM after the Predictor fix this? Like a Transformer-LSTM module. I wonder if the authors have tried this variant

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces the Slot-based Time-Space Transformer with Memory buffer (STATM), an object-centric learning model for videos. The model replaces the predictor in SAVi and SAVi++ with a spatiotemporal attention component that attends to slots in previous timesteps, which are stored in a memory buffer. The model is evaluated on the MOVi datasets, showing improvements in segmentation quality.

### Strengths
This paper tackles an important problem of improving object-centric learning models for videos. While spatiotemporal transformers have been applied to Slot Attention-based video models before [1], they have not been trained in an end-to-end fashion, as far as I know. The experimental results show improvements over SAVi and SAVi++, especially on the more complex MOVi-D and MOVi-E datasets, although I have concerns about these results that I state below.

[1] SlotFormer: Unsupervised Visual Dynamics Simulation with Object-Centric Models. https://arxiv.org/abs/2210.05861

### Weaknesses
There a few instances in the paper where the authors make statements that are not well-supported by their experiments. For example,

- The title and abstract emphasize reasoning, but this is not supported by any experiments. I think it is fine to use reasoning as a motivation for their model, but if the title includes “Reasoning-Enhanced”, I would have expected some experiments showing this ability.
- Similarly, I feel several statements connecting their model to human behavior are too strong and not supported by empirical evidence. For example, in the abstract, they state “We demonstrated that the improved deep learning model exhibits certain degree of rationality imitating human behavior”, but I don’t believe this is strongly supported by their experiments.
- In Section 4.1, “Essentially, higher prediction accuracy leads to better segmentation performance. If predictions are highly accurate, we don’t need to track objects at every step. Instead, we can focus on predicted locations, optimizing resource usage.” While the results do show STATM to perform better than SAVi, it is not clear that this is because of more accurate predictions. There is also no evidence of improved resource usage from STATM.

I also have a few issues with the experimental results section:

- The results for SAVi on MOVi-D and MOVi-E seem very low when compared to the SAVI++ paper (18.4 vs. 59.6 FG-ARI for MOVi-D and 10.8 vs. 55.3 FG-ARI for MOVi-E). I understand the authors only trained for 100k steps with a smaller batch size, but this indicates to me that the model is not well-trained yet. While we can draw some conclusions about the training speed from these results, I think we need to be careful about drawing broader conclusions, especially when the converged value (from the SAVI++ paper) is so much higher.
- The first ablation where we test on a smaller buffer length than we train on does not seem too informative. In this case, the model is being trained on a different setting than it was tested on so we can expect the performance to decrease.


Minor typos in Figure 1:

- Hits → Hints
- Preceive → Perceive
- Memery → Memory

### Questions
- Is there any positional encoding applied to the slots (either time-wise or slot-wise or both)? Otherwise, how are slots from different timesteps distinguished?
- In the introduction, the authors write “The prediction step in SAVi and SAVi++ is similar to human inference, but the predictor module in SAVi and SAVi++ is somewhat simplistic, as it relies solely on single-frame information from the current time step for prediction.” While it is true that the predictor in SAVi only uses the slots from the current time step, the slots themselves may contain information from previous timesteps (eg. velocity) since they are updated iteratively. One thing I would be curious about is if the representations in STATM differ from the representations in SAVi in that they may not need to include information such as velocity since this can be inferred by the spatiotemporal transformer. One way to test this would be to try to predict velocity from the slot representations.
- In Equation 4, I am a bit confused about the notation. Why are k_{i, 0} and k_{0, t} written separately outside the brackets?
- At the end of Section 4.2, the authors write "Currently, models without prompts have limited relevance to our objectives.” Prompting is also mentioned in the limitations section, but not discussed elsewhere in the paper. What is prompting in this context?
- For clarity, I would suggest only explaining T+S and CS in the main text since that is what is used in most of the experiments. The explanation of the other variants can go in the ablations section.
- This model seems like it would be especially beneficial in cases where an object that appears in the first frame is fully occluded at some intermediate frame and then reappears in a later frame. I think it would be informative to try some experiments exploring this (even in toy settings).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper tackles the problem of object-centric learning in the video setting. The authors proposed to leverage existing video-slot attention architecture and add additional spatial and temporal attention blocks between current slots and previous slots to better model the spatial-temporal correspondence between frames. The authors leveraged a memory buffer for all history slot information considered. The resulting model achieves state-of-the-art results and shows consistent improvements over prior architectures on commonly used video object-centric learning datasets (MOVi).

### Strengths
The authors proposed an intuitive model for better modeling the spatial-temporal consistency between object-centric representations during video object-centric learning. Compared with previous models which mainly used simple models like self-attention for linking information between frames, this design considers longer history and more complex attention between slots at different time steps. By only adding this module, we can observe consistent improvement over prior architecture and achieving new state-of-the-art.

### Weaknesses
[-] Despite the performance improvement of adding this STATM module, the quantitative results of the backbone models seem to be exhibiting a rather big gap on several datasets (e.g. SAVi and SAVi++ on MOVi-E compared with results reported from the SAVi++ paper [here](https://browse.arxiv.org/pdf/2206.07764.pdf)). This hinders the evaluation of this paper's contributions, the authors might want to clarify the experimental settings to make these results more convincing.

[-] Following the previous point and as mentioned by previous works, slot attention seems very susceptible to the random initialization provided. The authors should consider reporting the results of several trials with different random seeds for better result presentation.

### Questions
See the Weakness section.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents STATM (Slot-based Time-Space Transformer with Memory buffer), a method that extends slot-based scene representation methods SAVi and SAVi++ by incorporating a “memory buffer” or history of previous scenes, instead of a single scene, to update slot representations for objects in dynamic videos. STATM improves over smaller versions of SAVi and SAVi++ on the kubric benchmarks in terms of segmentation performance. Longer memory buffers help with segmentation performance particularly for the hardest dataset (MOVi-E in Kubric), both during training and at inference time. Several parallels between the STATM model and human cognition are drawn.

### Strengths
The paper is tackling an interesting problem (decomposing dynamic scenes / videos into objects), using a technique inspired by human cognition (people have working memory, and use motion cues to decompose scenes into objects). There is a large community of researchers interested in slot-based models, and this paper provides a general mechanism for how to improve slot-based scene representations by incorporating a history of previous time-steps.

The paper is reasonably easy to follow. The method is explained with sufficient detail to support reimplementation, and the figures explaining the method are helpful. The experimental results are presented clearly, and ablations over the memory buffer are sensible and help provide insight for how useful the memory is, and in which contexts. The qualitative segmentations are nice to see as well.

### Weaknesses
Overall, while the paper is tackling an established problem with a method inspired by human cognition, there are several issues that would need to be addressed before it could be impactful for the broader community. These can be generally grouped into:
* Adding comparisons to prior work that uses multiple history steps to infer segmentation masks
* Removing false statements about the SAVi/SAVi++ results
* Expanding explanation / background on SAVi for unfamiliar readers
* Toning down / removing incorrect / unverified statements about human cognition

## Adding comparisons to prior work that uses multiple history steps ##
There are now several papers that use multiple history steps with slot-based models to infer better segmentation masks. Three prominent ones are 
* Wu et al, ICLR 2023, SLOTFORMER: UNSUPERVISED VISUAL DYNAMICS SIMULATION WITH OBJECT-CENTRIC MODELS
* Zoran et al, ICCV 2021, PARTS: Unsupervised segmentation with slots, attention and independence maximization. 
* Chen et al, ECCV 2022, Unsupervised Segmentation in Real-World Images via Spelke Object Inference. 

These methods all use dynamic information from videos in order to do better scene segmentation, and are highly related to STATM. The first two also have open source implementations (within the SlotFormer codebase, I think) that could be run on the Kubric MOVi benchmarks.

## Removing false statements about the results ##
There are some major issues with the presentation of the results that need to be addressed during a rebuttal. The first concerns *why* the SAVi models were trained in the manner they were (for up to 5x *fewer* iterations, and half the batch size). I understand that memory requirements may have prevented a larger batch size from being possible, but the number of training iterations and / or learning rate should be changed to compensate. The paper does neither, and as a result, the SAVi / SAVi++ results are *significantly worse* than what was reported in the original paper. 

The paper states: “Due to constraints of computing resources, our models were trained for 100k steps with a batch size of 32, which differs from the official implementation of SAVi (small, 100k steps, batch size of 64) and SAVi++ (500k steps, batch size of 64). Nevertheless, under equivalent conditions, our models consistently outperform the original counterparts: e.g., STATM-SAVi (small, 100k steps, batch size of 32) achieves comparable performance to the official SAVi (large, 500k steps, batch size of 64) on MOVi-E datasets, while STATM-SAVi++ (100k steps, batch size of 32) performs comparably to SAVi++ (500k steps, batch size of 64)”

“STATM-SAVi (small, 100k steps, batch size of 32) achieves comparable performance to the official SAVi (large, 500k steps, batch size of 64) on MOVi-E datasets”
* This statement is false. Looking at the SAVi paper, on MOVi-E, SAVi obtains an mIoU of 30.7, while STATM-SAVi on the same dataset obtains an mIoU of 9.0. This is a 21+ point difference, not at all comparable.

“while STATM-SAVi++ (100k steps, batch size of 32) performs comparably to SAVi++ (500k steps, batch size of 64)”
* This statement is also false. From SAVi++ paper, the mIoU for MOVi-E is 47.1 vs. the 27.9 reported for STATM-SAVi++ in this paper, an almost 20 point difference. 

These are dangerously misleading statements. Furthermore, the fact that the “small” versions of the model are performing dramatically worse than the larger models calls into question whether the “small” versions of the SAVi model are just behaving completely differently from the larger ones. If more training completely erases the advantage of STATM, what does that mean for the scientific conclusions? How should the broader community interpret this?

Given that it should be possible to train the model for longer, even if at a smaller batch size, I do not believe that the STATM results will meaningfully inform the community unless the base SAVi models are trained for longer, better approaching the original performance.

“On the MOVi-E dataset, the segmentation capability of the AS structure is not as robust as that of the CS structure, but it still outperforms the baseline” 
* This is not true, at least based on comparisons to Table 1. The AS model is only doing better for FG-ARI, but is doing worse for mIoU.

## Expanding explanation of SAVi/SAVi++ ##

SAVi / SAVi++ are not explained enough for a reader not already familiar with the methods. The only mention of how these methods work is that there is a prediction and correction step, and that the correction step uses inputs to update the slots, and that STATM is replacing the prediction step. It would be helpful to further explain what the corrector module is doing, since the STATM method relies on this. However, the only information in the paper about this comes from the following two quotes:
* “The correction step uses inputs to update the slots”
* “... represents the slot information extracted from the corrector module of SAVi and SAVi++ at time t.”

This is not sufficient for a reader unfamiliar with SAVi to understand how the corrector module works, and what aspects of it are important for the STATM model.

## Toning down / removing incorrect / unverified statements about human cognition ##
The authors allude to human cognition throughout the paper as having inspired their architectural decisions for STATM. However, several statements made are either incorrect or unverified, and are therefore not helpful for bridging the cognitive science and machine learning communities. I list some examples below

“However, much like humans cannot predict the appearance of new objects in the next moment, the predictor faces similar limitations. At the initial moment, if the corrector cannot provide sufficiently accurate object information to the predictor, the predictor cannot offer precise prediction information for the corrector either.”
* These are not analogous situations, nor does it have anything to do with humans. No algorithm will be able to invent a correct appearance for an object if they do not have the sensor information for it. 

In the section “Ablation Experiment of Memory Module.”:
“This aligns with human learning habits. Gathering more information at once is more conducive to humans in recognizing and summarizing patterns.”
* This is not necessarily true. Famous examples include “one-shot learning” from Lake et al, 2015.

“The model’s tracking and segmentation capabilities over a duration equal to the training frames are less affected by memory (see Figure 4b). This is analogous to a scenario where a person has observed a significant amount of object motion in various scenarios over a time duration t. Subsequently, when asked to predict or describe how objects move within that t time duration, as long as the inquiry doesn’t extend beyond t, the person should still be able to provide reasonably accurate predictions and explanations, even if their view is obstructed or their memory is restricted.”
* I did not understand this argument. Many papers have shown that people can extrapolate physics beyond “trained” time-frames trivially. Some of the cited works in the paper demonstrate that (Smith et al, 2019, Ullman et al, 2017) for example. If this claim is important, it needs to be backed up with a cognitive science citation.

### Questions
* How does this model compare to alternatives that also use information in time to do segmentation (the baselines presented in “weaknesses”)?
* What happens to the advantage of STATM if the base SAVi / SAVi++ models are trained for longer, to compensate for the smaller batch size?
* What are the critical connections to human cognition that you would like to make, and what are some citations for those?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

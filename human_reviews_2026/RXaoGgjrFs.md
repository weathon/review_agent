# SlotFM: A Motion Foundation Model with Slot Attention for Diverse Downstream Tasks

- Decision: Reject
- Scores: 4, 6, 8

## Abstract
Wearable accelerometers are used for a wide range of applications, such as gesture recognition, gait analysis, and sports monitoring. Yet most existing foundation models focus primarily on classifying common daily activities such as locomotion and exercise, limiting their applicability to the broader range of tasks that rely on other signal characteristics. We present SlotFM, an accelerometer foundation model that generalizes across diverse downstream tasks. SlotFM uses Time-Frequency Slot Attention, an extension of Slot Attention that processes both time and frequency representations of the raw signals. It generates multiple small embeddings (slots), each capturing different signal components, enabling task-specific heads to focus on the most relevant parts of the data. We also introduce two loss regularizers that capture local structure and frequency patterns, which improve reconstruction of fine-grained details and helps the embeddings preserve task-relevant information. We evaluate SlotFM on 16 classification and regression downstream tasks that extend beyond standard human activity recognition. It outperforms existing self-supervised approaches on 13 of these tasks and achieves comparable results to the best performing approaches on the remaining tasks. On average, our method yields a 4.5% performance gain, demonstrating strong generalization for sensing foundation models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a time-frequency slot attention-based motion foundation model for accelerometers, aiming to generalize across multiple downstream tasks (HAR, pose, gesture, motion, etc.). The paper combines frequency domain decomposition with a slot-based self-attention structure and introduces two regularization losses, SSIM and MS-STFT, to improve signal reconstruction quality.

### Strengths
This paper is well organized. 
The overall logic is clear.

### Weaknesses
Overall, this is a reasonably good paper; however, while the motivation is to establish a motion foundation model that generalizes across multiple accelerometer tasks, the authors do not clearly explain why existing multi-task SSL models fail to generalize to the same task set. Furthermore, the theoretical basis for the slot attention mechanism, originally used for object decomposition in images, (such as the equivalence of time-frequency structures) for its transfer to time-series signals is not sufficiently demonstrated. A clear hypothesis or quantitative motivation is lacking to explain why the "slot-based" method has a natural advantage in IMU signals.

Weaknesses:
1.	The paper only pre-trained on CAPTURE-24, while the test tasks differed greatly from the original domain. Although the results showed good generalization, there were no explicit cross-domain validation (cross-person or cross-device) experiments.
2.	Reporting only the average decrease percentage in ablation experiments is insufficient to validate the independent contribution of the modules. It is recommended to include interpretability visualizations or t-SNE results to demonstrate the separability of slots across different tasks.
3.	Although the authors claim to have open-sourced the code and task settings, they haven't provided details on hyperparameters in the paper, nor have they included training time, memory usage, or training stability reports for comparative experiments in the appendix. In the experimental setup, SlotFM uses a fixed number of slots, a self-attention head, and a three-way ResNet encoder, making the number of parameters seem likely to be significantly larger than the baseline (vs. single ResNet). While the authors state that the total number of parameters is "similar," they haven't provided a precise parameter comparison table. Considering the model requires 500 epochs of training on a 4×V100, the actual computational cost is high, potentially impacting reproducibility and practical applications.

### Questions
1.	Slot Attention's cross-attention is insensitive to the input order, so in this work the authors introduce 2D positional embedding, but do not prove that this positional encoding can preserve the temporal dependence of continuous signals.
2.	Since frequency band decomposition has already separated low/high frequency information, is the gain of Slot Attention in reallocating slots significant, or is there any redundancy?
3.	The loss function is defined as a combination of three loss functions (MSE, SSIM, MS-STFT). However, SSIM is essentially a similarity measure (range 0-1), while MSE and STFT loss appear to be unbounded energy errors, with a huge difference in magnitude. The authors set β=100 and γ=0.1 based solely on experience without providing scale normalization or sensitivity analysis. Applying such a large difference in loss combination may also lead to insufficient generalization ability of the method, resulting in optimization instability or bias towards specific frequency bands, making it extremely difficult to apply this method to other fields or different datasets.

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
5

### Summary
This paper presents a novel foundation model for a wide variety of motion related tasks with sota performance. Additionally, they proposed a novel time-frequency slot attention methodology adapted for time-series ssl that exploits specific time and frequency information.

### Strengths
* Well formulated time-frequency slot attention mechanism that exploit time-series specific properties for the setting.
* Thoughtful loss regularizers that address key problems
* Slot reconstruction visualization offers interpretable insights into specific learned slots
* Diverse downstream tasks and datasets that cover a wide range of motion-related tasks, greatly expanding upon prior work

### Weaknesses
* The model architecture in 3.2 is not super clear, and generally requires that readers have an exact understanding of slot attention. as another example, a GRU cell is mentioned, but not seen in Figure 3.  the paper could perhaps benefit from including a mathematical formulation of architecture. this is not a strict necessity, but something to improve clarity in 3.2 would be helpful.
* Figure 4 with weights per slot distribution is not well explained. Why are slots 4/5 the beginning of the signal? This analysis on slot weights seems underexplored and underexplained.

### Questions
* This is not a strong criticism, but something I would appreciate further discussion on is needing to calculate multi-scale STFT for every instance and the time complexity associated with this. this seems to me to be very slow, thus making this method not necessarily scalable for large-scale pre-training. once again, this does not kill the idea of the paper, but i would appreciate better understanding of the limitations in general. 
* Haresumadram et al has not released their exact train/val/test splits for all datasets and giving the numbers are directly quoted from that paper, this should be addressed more clearly. The RelCon codebase does have the splits for some of the datasets, but notably not FoG nor USC-HAD. How were the exact splits determined? If they were not exactly the same, I recommend reaching out to Haresumadram to obtain the exact splits during rebuttal or final version.  
* A code release is mentioned, but it is not a part of the current submission. Can this be clarified?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a foundation model for accelerometer tasks called SlotFM that is able to generalize across diverse downstream tasks through self-supervised learning.

It exploits Slot Attention to encode each signal into multiple vectors or slots across several time intervals and three frequency bands, which are used to reconstruct the input signal during training. These slots are used during inference as embeddings for accomplishing downstream tasks. The authors introduce two regularizers to capture local structures and frequency patterns which encourage preservation of the fine-grained details during reconstruction.

The foundation model achieved state-of-the-art results on 13 downstream tasks, and almost-top results for another 3 tasks, showing the ubiquity of the learned slots for diverse range of problems. Surprisingly, in almost all cases, it beat or is just as good as models trained by supervised learning on smaller datasets

### Strengths
**Significance**
* Experiments show that self-supervised foundation models can learn embeddings which are useful for a large variety of downstream accelerometer tasks.

**Originality**
* As far as I can tell, this is the first time Slot Attention (originally for computer vision) is applied to accelerometer data
* Improved reconstruction results from two novel regularizers

**Clarity**
* The model architecture and experimental methods are clearly described

**Quality**
* State-of-the-art results across almost all self-supervised methods
* Almost state-of-the-art results when compared with supervised methods

### Weaknesses
* The iterative slot attention mechanism is described in both Section 3.1 and 3.2. Even then, it is hard to imagine how this mechanism is actually implemented without reading the original slot attention paper. I am also confused about the implementation of signal reconstruction during pretraining that is describe at the end of Section 3.2. Perhaps the pseudocodes can be provided in the Appendix.

* The weights per slot in Fig 4 seem quite uniform across slots to me, for all tasks except Writing. I do not think they warrant the discussion at the end of Section 5.1,  "The weight distribution among slots differs between tasks, with some tasks emphasizing particular slots while others assigning similar weights to all slots," nor does it warrant the observation at the end of the Introduction, "Finally, analysis of head weights shows that certain slots are emphasized more than others depending on the task, demonstrating the adaptability of slot-based embeddings." More empirical proof is needed, or the statements should be toned down.

### Questions
1. Strangely, the task accuracies do not scale with the number of frequency bands or the number of temporal intervals, probably because of overfitting. This seems to suggest that some kind of regularization can help. Do you have any insight/analysis of what is going on here?

2. From Fig 2, it seems that there are 3 x 8 = 24 "slots" (spatial-temporally) but in Fig 4, there are only 8 weights for 8 slots (temporally). Do you consider the combination of low-middle-high band components in one time-interval as one slot? Why not allow the weighted sum of the band components for downstream tasks, so we have 24 weights rather than 8?

### Soundness
3

### Presentation
3

### Contribution
3

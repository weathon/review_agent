# Toward Generalizable Deblurring: Leveraging Massive Blur Priors with Linear Attention for Real-World Scenarios

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 4, 2

## Abstract
Image deblurring has advanced rapidly with deep learning, yet most methods exhibit poor generalization beyond their training datasets, with performance dropping significantly in real-world scenarios. Our analysis shows this limitation stems from two factors: datasets face an inherent trade-off between realism and coverage of diverse blur patterns, and algorithmic designs remain restrictive, as pixel-wise losses drive models toward local detail recovery while overlooking structural and semantic consistency, whereas diffusion-based approaches, though perceptually strong, still fail to generalize when trained on narrow datasets with simplistic strategies. Through systematic investigation, we identify blur pattern diversity as the decisive factor for robust generalization and propose Blur Pattern Pretraining (BBP), which acquires blur priors from simulation datasets and transfers them through joint fine-tuning on real data. We further introduce Motion and Semantic Guidance (MoSeG) to strengthen blur priors under severe degradation, and integrate it into GLOWDeblur, a Generalizable reaLwOrld lightWeight Deblur model that combines convolution-based pre-reconstruction & domain alignment module with a lightweight diffusion backbone. Extensive experiments on six widely-used benchmarks and two real-world datasets validate our approach, confirming the importance of blur priors for robust generalization and demonstrating that the lightweight design of GLOWDeblur ensures practicality in real-world applications.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes a generalizable approach to real-world image deblurring. Observing that existing models suffer significant performance degradation on unseen data, the authors identify the core issue as the lack of diversity in blur patterns within training datasets, as well as an overreliance on pixel-wise losses that ignore structural and semantic consistency. To address this, they introduce a Blur Pattern Pretraining (BBP) strategy, which first learns blur priors from simulated datasets and then performs joint fine-tuning on real-captured data. They also present GLOWDeblur, a lightweight diffusion-based deblurring model that incorporates motion and semantic guidance to enhance restoration under severe blur. Experimental results across multiple benchmark and real-world datasets demonstrate the method’s superior generalization capabilities, highlighting the importance of blur priors and architectural design for practical deployment.

### Strengths
1.  Precise Problem Definition with Rigorous Evidence: The paper's motivation is clear and highly persuasive. It moves beyond the conventional discussion of "realism" by skillfully combining Table 1 (performance degradation) and Figure 3 (imbalanced pattern distributions). This robustly proves that the root cause of generalization failure is the biased and imbalanced distribution of "blur patterns" in the training data. 

2.  Shifting Research Focus with a Systematic Strategy: Based on this key insight, the paper successfully shifts the research focus from pursuing "realism" to pursuing "blur pattern diversity." It proposes a systematic data strategy to address this issue. It first learns blur priors from simulation datasets with broad pattern coverage and then jointly fine-tunes on real-world datasets to align the distribution.

3. Superior Visual Results: This strategy directly targets the core of the problem, and its effectiveness is intuitively validated by the superior visual results.

### Weaknesses
1. Weak Methodological Innovation and Unaddressed Concerns: The paper's novelty is limited, primarily relying on stacking existing modules like MoG and SeG. These additions also introduce significant concerns: the accuracy of MoG's motion estimation is unverified and risks misguiding the restoration. Furthermore, the usage of SeG is ambiguous; if a VLM is required at test time, it introduces an unfair external annotation, and its robustness on low-quality, poorly-described images is unexplored.

2. Contradictory Metrics and Fidelity Concerns: The experimental results show a clear contradiction. The method excels in No-Reference (NR-IQA) metrics like MANIQA, yet it significantly lags behind in traditional Full-Reference (FR-IQA) metrics such as PSNR and SSIM. This strongly suggests that the model sacrifices fidelity for perceptual quality, likely suffering from the common "hallucination" problem in diffusion models (generating plausible but inaccurate details). The critical omission of the LPIPS metric fails to resolve these fidelity concerns.

3. Misleading "Lightweight" Claim: The paper's claim of being "lightweight" is misleading. It selectively highlights optimizations in the diffusion core while ignoring the total system complexity (including the UNet, MoG, and potentially a large VLM for SeG). The lack of FLOPs or actual inference time figures, combined with the mention of high-end training hardware (8x A800 80G GPUs), makes the claims of efficiency highly questionable.

### Questions
1.  Is the SeG module required to run during the inference stage? If so, does this imply the model relies on a large external VLM to achieve its advantage, leading to an unfair comparison and contradicting the "lightweight" claim? 

2.  Is the motion estimation within the MoG module trained end-to-end, or does it use a fixed pre-trained model? If the former, are there specific loss functions or visualizations to prove it generates correct motion guidance? If the latter, how is its accuracy and applicability to your dataset ensured? 

3.  How do the authors explain the significant gap where the model lags in PSNR/SSIM but excels in NR-IQA metrics? Does this imply the model suffers from the common "hallucination" issue (deviating from ground truth)? Why is the critical LPIPS metric missing to validate the perceptual fidelity against the ground truth? 

4.  To validate the "lightweight" claim, what are the average inference time, the number of the parameters, and the FLOPs for the proposed method? 

5.  Could you please detail the collection and filtering criteria for the self-created RWBlur400 dataset? Does this dataset have corresponding ground truth? If not, is relying solely on NR-IQA metrics sufficient to claim robust real-world performance?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
1. This paper discussed the importance of training deblurring methods with diverse blur patterns, and proposed a training strategy that first pre-trains the model on diverse synthetic blur pattern then real blurred imags. 
2. A novel deblurring architecture consists of lighweight diffusion model is also proposed.

### Strengths
1. Performance shows that the proposed training strategy is helpful.
2. The lightweight diffusion architecture is novel, which is a good tradeoff between generalization and efficiency,

### Weaknesses
1. I think the paper should specifies that it is proposed to handling motion blur instead of general blurs.
2. The training pipeline is complex and hard to reproduce.
3. No explicit inference time evaluation.

### Questions
1. In line 144 and 129, I guess BBP is not the abbreviation of Blur pattern pretraining
2. How to get the blur pattern statistics shown in Table 3.
3. Were captions generated on degraded images or sharp images? If it is the latter, how to get correct captions during inference?
4. Is such an aggressive encoder able to reconstruct original images?
5. I know authors claimed that mix-training is sub-optimal, but I wonder the performance of training the model with just one step using all datasets mentioned in this paper.
6. In the owl of Figure 6, I wonder if the feather on the wing is really the reconstructed details or just hallucination. If it is hallucination, then it is dangerous to use it in camera.
7. How were other compared methods trained, on the same dataset or simply gran the pre-trained ckpt?
8. If the architecture is one of the contribution, then even without the BPP it should performs better than other methods. I would like to see more results about this. 
9. Any evaluations of inference time?

### Soundness
3

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
3

### Summary
This paper addresses the limited generalization of image deblurring models in real-world scenario. It identifies main challenges from blur diversity-realism tradeoff and proposes BPP to learn diverse blur priors from simulated data, transferring them to real domains via joint fine-tuning. To further improve robustness, the authors introduce Motion and Semantic Guidance (MoSeG), which leverages motion cues and semantic context to enhance restoration quality.

### Strengths
1. Writing: The paper is well-organized and clearly written, making it easy to follow.

2. Logical Idea: The integration of motion-related and semantic-related information for deblurring is intuitive and well-motivated, representing a logical extension of existing approaches. The overall framework design is cohesive, and experiments demonstrate promising results.

### Weaknesses
1. Motion Guidance: The proposed motion modeling appears limited to 2D directional blur, whereas real-world blur often includes depth-axis motion components. Consequently, BPP may fail to capture full 3D motion complexity, reducing its applicability to realistic scenarios. Moreover, since motion trajectories require paired sharp images to be computed, the motion guidance component can only be trained on synthetic datasets, potentially restricting its generalization to real-world data.

2. Semantic Guidance: Under conditions of severe blur, the pretrained VLM~(QwenVL) model used for semantic extraction may produce inaccurate or unreliable outputs, weakening its contribution to deblurring quality.

3. Typo: BBP -> BPP

### Questions
1. Could the authors provide a comparison with models trained on all major datasets (GoPro, HIDE, REDS, RealBlur, etc.) to more convincingly demonstrate the effectiveness of their approach?

2. What is the patch size or spatial scale used for motion estimation in motion guidance, and how sensitive is the performance to this choice?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper tackles the generalization problem in image deblurring caused by the biases in deblurring datasets. The authors identified the characteristics in the current datasets and investigated the generalization performance using the existing deblurring method, Restormer. Then, this paper proposes Blur Pattern Pretraining to use a simulation-based dataset, GSBlur, to pretrain the deblurring models, thus improving the generalization ability. Furthermore, the deblurring performance is enhanced by the proposed GLOWDeblur model, which consists of several auxiliary tasks like motion estimation, text-guided diffusion.

### Strengths
- The paper is well-motivated. The generalization problem in the image deblurring task is crucial and has not been fully explored.
- The characteristic analysis of the current deblurring datasets contributes to the deblurring community.
- The paper is well-written and easy to follow.

### Weaknesses
- There are concerns for the BBP in tackling the generalization problem: BBP relies on GSBlur, a larger existing simulation dataset. Thus, the improvement may be due to 1) GSBlur is a larger training dataset and 2) GSBlur covers the blur characteristics of each test set, rather than generalizing to new blur characteristics.
- GLOWDeblur significantly degrades the PSNR and SSIM in most cases in Table 3. However, these are major metrics in the image deblurring task. This method may severely affect the pixel-level similarity between the output and the ground truth, which is not a satisfactory deblurring result.
- Many auxiliary tasks (motion estimation, text-guided diffusion generation) are equipped into the model. This raises the concern of the efficiency compared with other methods. But this is not mentioned in the paper.

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
3

### Contribution
2

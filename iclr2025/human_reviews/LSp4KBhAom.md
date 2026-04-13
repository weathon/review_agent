## Human Reviewer 1

### Summary
This paper presents a method for self-calibrating 3D geometric models. The authors utilize a pre-trained model to generate predicted point maps and their corresponding confidence scores for input images. To automatically identify and mitigate overconfident predictions that may adversely affect fine-tuning performance, they re-parameterize the confidence term as an optimizable weight. This approach effectively reduces the influence of multi-view inconsistent predictions during the optimization process. Ultimately, the refined point maps and calibrated confidence scores serve as pseudo-labels for fine-tuning the pre-trained DUSt3R model using low-rank adaptation (LoRA) techniques. The authors achieve performance enhancements while maintaining an acceptable increase in computational costs.

### Strengths
1)	The paper has good writing and is easy to follow.
2)	The authors present a simple but effective pipeline to take advantage of the LoRA technique to improve performance of the 3D foundation model in a self-calibration manner.

### Weaknesses
The paper does not present major weaknesses.

### Questions
1)	Does Figure 3 solely visualize the prediction confidence and error map of the original DUSt3R method? The caption for Figure 3 should be clarified to reduce ambiguity. Figures 3 and 4 are complementary but they are on separate pages. Readers cannot quickly grasp the points of Figure 3 (a) (b) (c).
2)	Minor issues: missing references in Line 817.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This submission proposes a method to finetune large 3D models such as Duster on individual scenes. The finetuned parameters include camera intrinsics, extrinsics, point maps, and scales. To avoid finetuning a large 3D model directly, the method proposes to finetune them with LoRA. As a result, the proposed method improves per-scene reconstruction quality and camera estimation accuracy in an efficient way.

In terms of weaknesses, my primary concerns are: 1) missing a straightforward global optimisation baseline that is also initialised from Duster predictions, and 2) unconvincing comparison with COLMAP and similar works.

Overall, the paper offers an efficient approach to fine-tuning Duster for scene-specific applications, demonstrating notable gains in reconstruction and camera estimation accuracy. However, the unique advantages of this pipeline are not fully clear, especially given the missing global optimization baseline and the somewhat limited comparisons with similar works like COLMAP.

My initial recommendation leans toward a borderline reject, but I am open to adjusting my rating based on further discussion.

### Strengths
The paper presents an efficient method to finetune Duster for individual scenes, showing considerable improvements in terms of reconstruction and camera estimation accuracy.

### Weaknesses
***Missing baseline.***
A major weakness to me is missing a straightforward baseline, which is running a global optimisation that is similar to classical global bundle adjustment for all camera intrinsics, extrinsics, point maps, and scales, given initialisations from Dusters predictions. This baseline would also avoid finetuning Duster weights. I am curious about a fair comparison with this baseline, in terms of optimisation time, and per-scene optimisation accuracy.

***The comparison with COLMAP and similar works is not convincing.***
The proposed DUSt3R-Self-Calib always has an initial solution from Duster. It would be more fair to other works if they started from Duster results too.

***Writing (not affecting rating).***
* The text between Sec 5 and Sec 5.1 should be better sectioned and formatted. Currently, tasks and baselines are mixed together.
* In Eq. 7, the notation "W" is reused. It's already used for image size HW.

### Questions
1. In Fig 3 and Fig 4, how do Duster predicted confidence maps look like?
2. Fig 3d: this fig shows calibrated confidences are smaller than predictions. Two questions here:
    1) Is it always desirable (better) to have a smaller calibrated confidence compared to Duster predicted confidence?
    2) is there any example where the calibrated confidence is larger than Duster predicted confidence, but still leads to good finetuning results in terms of self-calibration and 3d reconstruction accuracy?
3. Writing (not affecting rating):
    1) L181, "... the global point maps can be further re-parameterized via back-propagation" -- What does “re-parameterized via back-propagation" mean?
    2) L222, "second regularization term" -- it seems like there is only one reg term?

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
6

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper presents an effective and straightforward calibration method based on LoRA to fine-tune a 3D foundation model for specialization in a specific scene, achieving a notable performance increase of over 80%. Extensive experiments conducted across multiple datasets and downstream tasks, along with comparisons to various baselines, further validate the model's performance.

### Strengths
This task is quite intriguing, presenting an efficient and useful technique for calibrating pre-trained 3D models for specific 3D tasks. 
The concept of calibration used to be employed in the field of uncertainty estimation to enhance output quality. Bringing calibration into a 3D foundation model presents a novel approach for "making foundation models" perform more effectively or align better with new or specialized tasks.

The techniques sound and are easy to follow. Despite some poor observations (e.g., low light), there are still sufficient quality observations from the same scene that can provide valuable constraints and useful signals to supervise the calibration process. 

The focus on confidence is both interesting and beneficial, as it addresses ambiguity and enhances accuracy. Confidence metrics effectively aid in selecting precise components for fine-tuning during experiments, even facilitating dynamic scenarios.

The use of M-estimation for optimization, as opposed to traditional gradient descent, is another noteworthy aspect.

It is encouraging to see further extensions to Mask3R, which demonstrate the generalizability of this method.

The paper provides good details for reproducibility.

The experiments are extensive, covering various 3D downstream tasks and comparing results with state-of-the-art methods in each category. The results indicate that close to “fine-tuning” level performance can be achieved with minimal calibration.

### Weaknesses
The system is designed to adapt to a single scene, which appears to be a limitation of the DUSt3R approach.

To evaluate performance, the methodology requires half of the test split data (1,000 images out of 2,000), which may limit practical applications. I am curious about the results if only 5%, 10%, or 30% of the data were available for calibration. Furthermore, it remains unclear what portion of the calibrated images comes from the TUM and Waymo datasets.

Further details are needed regarding the selection of the 161 test scenes. Is this selection consistent with common practices in previously published baselines?

From Figure 6, it is evident that performance before calibration is already visually impressive, with significant improvements primarily noted in the corner regions. While the reported metrics are impressive, it remains unclear how the calibration contributes to more visually sensible performance.

### Questions
Please refer to the questions outlined in the weaknesses section, which includes an ablation study on the portion of data used for calibration, details on how the scenes were selected, and clarification on the visually sensible improvements observed.

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
2

---

## Human Reviewer 4

### Summary
The paper proposes an approach for calibration 3D foundation model, particularly DUSt3R and MASt3R. Given a small sample of images, the paper extracts the most multi-view consistent 3D points (through a robust global optimisation and confidence calibration) and then feeds them to a LoRA-based fine-tune process.

### Strengths
The paper is very well written and easy to understand.

I very much appreciate the core calibration framework text section and underlying method, in that is it technically principled, produces strong state-of-the-art improvements, and is presented in an intuitive manor. This is a great example of how ML/CV papers should be written.

It is the first time I see LoRA used in the context of a 3D foundation model. I expect this to help greatly increase the potential impact of the paper.

The overall approach produces significantly improved results over the DUST3R baseline, and the small sizes of the LoRA could potentially lead to practical applications.

### Weaknesses
I find little weakness with the paper. Mainly, I would have liked to see more experiments on MASt3R in the body of the main work, and perhaps an inclusion of the calibrated-confidence-based pseudo labelling in the MASt3R integration. This is perhaps the only point that stops me from recommending a strong accept.

### Questions
-

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
8

### Confidence
4
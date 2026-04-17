# Learning from Noisy Preferences: A Semi-Supervised Learning Approach to Direct Preference Optimization

- Decision: Accept (Poster)
- Scores: 6, 6, 2

## Abstract
Human visual preferences are inherently multi-dimensional, encompassing aesthetics, detail fidelity, and semantic alignment. However, existing datasets provide only single, holistic annotations, resulting in severe label noise—images that excel in some dimensions but are deficient in others are simply marked as winner or loser. We theoretically demonstrate that compressing multi-dimensional preferences into binary labels generates conflicting gradient signals that misguide Diffusion Direct Preference Optimization (DPO). To address this, we propose Semi-DPO, a semi-supervised approach that treats consistent pairs as clean labeled data and conflicting ones as noisy unlabeled data. Our method starts by training on a consensus-filtered clean subset, then uses this model as an implicit classifier to generate pseudo-labels for the noisy set for iterative refinement. Experimental results demonstrate that Semi-DPO achieves state-of-the-art performance and significantly improves alignment with complex human preferences, without requiring additional human annotation or explicit reward models during training.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper is about generating T2I results that can incorporate human preferences by going beyond of collapsing the multi-dimensional preference to a single binary indicator. The paper proposed a method that by recognizing human preferences are of multi-dimensional and noisy (or in other words, emphasizing various aesthetics perspectives). The method is to divide a training set into clean and noisy subsets and start the training of a model with the clean subsets. The model then moves on to the noisy preference subset to train iteratively to align the model's generation with human preference.

### Strengths
Strengths of the work is the recognition of the noisy nature of human preference in assess AI-generated images. Building upon this strength is the strength of highlighting the drawback of simply classifying a generated image as winner or loser as do so would collapse the original multi-dimensional human preference to an overly too simple binary criterion. Another strength is the division of a training set to clean and noisy subsets and the design of method that iteratively trains on the noisy subset to align the model's output iteratively with human preference.

### Weaknesses
It is not clear, given a training set, when it is divided into clean and noisy preference subsets along dimension k, does it mean for a different dimension, the clean and noisy subsets will be different.
It is unclear how the loss function of Eq. (8) was derived. It seems from the beginning of the paper, the iterative training only applies to the noisy preference subset, which is understandable, but then why for iterative refinement step, Eq. (8) includes L_labeled? Shouldn't Eq. (8) only need to be concerned with L_unlabeled^i for each iteration i?
Contrary to what the paper claimed in line 418, it seems there was not big improvement from iter 1 to iter 2 in ablation study. It seems that the improvement came from iter 0 to iter 1. From this perspective, it would likely suggest that iterative refinement is not necessary.

### Questions
Please see weaknesses.
It would be helpful if the authors could give examples of "noisy labels", as well as some examples of multi-dimensional human preferences.
Does iter 0 mean no refinement and only the clean subset was used?

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper points out a key issue faced by the current image preference alignment methods based on Diffusion-DPO: human visual preferences are inherently multi-dimensional (such as composition, details, semantic alignment, aesthetics, etc.), but existing preference datasets only provide a single "overall winner" label, leading to significant label noise. The authors theoretically analyze that compressing such multi-dimensional preferences into binary labels will cause conflicting gradient signals, thereby hindering model optimization. To address this problem, the authors propose Semi-DPO, a new method that reconfigures preference alignment as a semi-supervised learning problem. This method first uses the consensus of multiple pre-trained reward models to filter out "clean" preference pairs as labeled data, and the rest are regarded as noisy unlabeled data. Then, through iterative self-training, it uses the model itself as an implicit classifier to generate fine-grained pseudo-labels at different time steps of the diffusion process, thereby decoupling conflicting signals. Experiments show that Semi-DPO significantly outperforms existing methods on multiple metrics, reaching state-of-the-art (SOTA) levels, without the need for additional manual annotations or explicit reward models.

### Strengths
1. The theoretical analysis is solid: through the derivation of the lower bound of gradient variance, it rigorously proves how dimension conflicts lead to training instability, providing theoretical support for method design.

2. The method design is ingenious: combining the DPO framework with semi-supervised learning, it utilizes the inherent discriminative ability of the diffusion model in DPO training to generate time-step conditional pseudo-labels without modifying the model architecture.

### Weaknesses
1. The method relies on multiple rounds of self-training (2–3 rounds are used in the paper) and multi-reward model consensus filtering, making the training process more complex and time-consuming than standard DPO (see Appendix 6.4), which may limit its application in resource-constrained scenarios.

2. The initial clean set only accounts for approximately 21% of the original data. If the consensus reward model itself has systematic biases (e.g., poor performance on certain prompt types), it may affect the quality of cold start.

### Questions
NA

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors argue that compressing multi-dimensional human preferences into binary labels creates conflicting gradient signals during Diffusion-DPO training. They propose a two-stage method: 1) “Multi-Reward Consensus” filters a dataset into clean and noisy subsets based on the unanimous agreement of five pre-trained reward models, and 2) “Iterative Self-Training” uses a model trained on the clean set to generate timestep-conditional pseudo-labels for the noisy set, which are then used for further training. Experiments on Pick-a-Pic V2 show improvements over Diffusion-DPO and Diffusion-KTO baselines.

### Strengths
1. The Semi-DPO framework is a clever and practical solution: Using a committee of diverse reward models for robust data partitioning. Leveraging the diffusion model itself as an implicit, timestep-wise preference classifier. The dynamic, timestep-conditional thresholding for pseudo-label selection, which accounts for the varying reliability of the model across the denoising process.
2. The paper excels at identifying and formalizing a fundamental issue in preference learning for generative models. The argument that multi-dimensional preferences are a source of label noise is well-illustrated.

### Weaknesses
1. The innovation is quite limited, and I believe the “Multi-Reward Consensus” in the first stage is very similar to CaPO. It involves using multiple rewards for evaluation and selecting samples where the winner outperforms the loser across all dimensions for learning. Additionally, the idea of phased training is somewhat similar to [5], relabel is somewhat similar to [8].
2. There is a lack of comparison with many baselines.  like [1]SPO(SD1.5 SDXL), [2]DDIM-InPO(SD1.5 SDXL), [3]Flow-GRPO(SD3.5-Medium) [6]DSPO (SD1.5) [7] MaPO(SD1.5 SDXL). These baselines (checkpoints) are open-sourced.
 
   [1]Aesthetic Post-Training Diffusion Models from Generic Preferences with Step-by-step Preference Optimization. CVPR2025

   [2]InPO: Inversion Preference Optimization with Reparametrized DDIM for Efficient Diffusion Model Alignment. CVPR2025

   [3]Flow-GRPO: Training Flow Matching Models via Online RL

   [4]Calibrated Multi-Preference Optimization for Aligning Diffusion Models. CVPR2025

   [5]Curriculum Direct Preference Optimization for Diffusion and Consistency Models. CVPR2025

   [6] DSPO: Direct Score Preference Optimization for Diffusion Model Alignment. ICLR2025

   [7] Margin-aware Preference Optimization for Aligning Diffusion Models without Reference

   [8] Smoothed Preference Optimization via ReNoise Inversion for Aligning Diffusion Models with Varied Human Preferences. ICML 2025

### Questions
1. The SSL literature has many techniques for handling noisy labels (e.g., co-teaching, robust losses). Why was the self-training/pseudo-labeling paradigm chosen over others?
2. Can further comparative experiments be conducted on T2ICompbench?
3. Is the training time, memory footprint, or inference time comparable?
4. Are there any failure modes observed?
5. How does your model perform in controlled generation tasks, and could it lead to the loss of certain properties?
The final score is contingent upon the following revisions: For score 4 :supplemental baseline comparison mentioned，For score 6: Provide experimental efficiency metrics and comparative results on the T2ICompBench.

### Soundness
2

### Presentation
2

### Contribution
2

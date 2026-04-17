# The Robustness-Security Paradox: Channel-Aware Feature Learning for Adversarial Watermark Exploitation

- Decision: Reject
- Scores: 6, 2, 4

## Abstract
Watermarking is crucial for establishing provenance and detecting AI-generated content. While current approaches prioritize robustness against real-world distortions, we explore how the robustness-security tradeoff manifests in deep learning-based watermarks: robust watermarks necessarily increase the redundancy of detectable watermark patterns embedded in images, creating exploitable information leakage. Leveraging this insight, we introduce an attack framework that extracts watermark pattern leakage through multi-channel feature learning using pre-trained vision models. Unlike previous approaches that require extensive data or detector access, our method achieves both watermark removal (detection evasion) and watermark forgery attacks with just a single watermarked image in a no-box setting. Extensive experiments demonstrate our method outperforms state-of-the-art techniques by 74\% in detection evasion rate and 47\% in forgery accuracy, while preserving visual quality.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents a watermark removal and a watermark forgery attack against post-processing watermarking methods. Their method is based on the key observation that the watermarking signal embedded by these watermarking methods is usually constrained to a few feature channels. By identifying these channels through clustering, they can optimize noise that, when added to the image, either mitigates the watermark signal (for removal) or replicates it (for forgery). They evaluate their method against five post-processing watermarking methods and find overall improvements in success rate and visual quality degradation.

### Strengths
1. Well-motivated and well-written.
2. Solid methodology and approach.
3. Good experimentation section

### Weaknesses
1. Overreaching claims. You only consider post-processing watermarking in your study, as stated in Section 3. However, throughout the paper, you make claims that your results apply to ALL watermarking methods (which would include in-processing watermarks). Examples of general claims for which you only give evidence for post-processing watermarks: line 215, lines 249-251 (the paradox could be limited to post-processing watermarks), line 395 (“representative watermark algorithms”, not true, it would be representative post-processing watermark algorithms”; “universally” overreaching, you did not prove that and only gave evidence for post-processing watermark methods). Also, as per your study against the Tree-Ring watermark, you find that your method actually does not perform well against in-process watermarks, invalidating your universality claim. Your abstract and conclusion should also make it clear that you are only considering post-processing watermarks. As they are, they imply all types of watermarks, which is not the case. These overreaching claims need to be addressed.

2. Misleading results. You claim in the abstract that you improve the state-of-the-art by 74% for watermark removal, which is not what you find in your evaluation (41.8% and 64%). I don't know where the 74% is coming from. Also, the way you computed your improvements over previous work is not fair. Computing your improvement over the best attack for each defense would be a better way of showing how you improve against state-of-the-art in practice. Also, for forgery, computing the overall average and claiming it to be your improvement over state-of-the-art in the abstract is misleading since some of the methods perform quite poorly. You improve over the second-best method by 10% on average, which is still a good improvement. You, however, cannot claim you improve over state-of-the-art by 47% (as you do in the abstract); that is not true. The results reported need to properly reflect actual improvement against state-of-the-art; average improvement against all defenses studied is not a great metric, as it favors including poorly performing attacks like Gaussian noise and Gaussian blur to artificially inflate improvement, which are not state-of-the-art attacks. These results need to be corrected throughout the paper, especially in the abstract and conclusion, as they are misleading regarding the actual improvements.
3. The epsilon values selected are quite large. While for removal 0.08~20/255 is large, it’s still somewhat manageable (although, usually, for the l-infinity norm, for high resolution images, adversarial attack research has usually used epsilons like 4/255 [1]). However, for forgery, the upper bound on the noise is 0.143 ~ 36/255, which is a lot
. You also don’t include any images in the main part of the paper. For your ablation study, you further increase epsilon 2, which raises the upper bound on the forgery noise to 0.158 ~ 40/256. This is a significant amount of noise.

4. The False Positive Rate (FPR) used to calibrate the thresholds is not stated; depending on the value, it could affect the success of either attack.

[1] Croce, Francesco, et al. "Robustbench: a standardized adversarial robustness benchmark." arXiv preprint arXiv:2010.09670 (2020).

### Questions
Questions:
1. What process did you use to embed watermarks for your feasibility study? It would be good to specify.
2. What is the loss function $\mathcal{L}$ you use for your attacks? You didn’t specify in Section 4.2, and in Section 5.1, Parameter Settings, you mention the SSIM and L1/L2 norms. Are these the loss functions you use? If so, how does SSIM work over channel features?
3. What do k and n stand for in the K-Means algorithm?
4. How was your feature extractor trained, more specifically, on what data?
5. How did you decide on the epsilon values you selected? Have you tried varying them, in particular, lowering them? 
6. For the forgery attack, you specify that the forged attack is obtained as one of two possible images (either only delta is subtracted or delta is subtracted and delta s is added), which one do you use when measuring the success rate? Are your results the average of both, or are you only picking the best of the two? If it’s the latter, it’s not a fair comparison to other methods that only output one forgery, since you essentially get two tries.



Suggestions:
1. AIGC, as an acronym, is not explained.
2. This is both a question and a suggestion to satiate my curiosity. For the watermark removal attack, did you try adding an extra loss component to minimize the effect of the added noise on the feature channels not selected by your clustering algorithm? It could help preserve image quality and focus the perturbations on the selected feature channels.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes a watermark attack method that achieves both watermark removal and forgery. The proposed method is evaluated against several existing watermark-generation processes and compares with several existing attack methods. Substantial improvement is demonstrated.

### Strengths
The research question is timely and interesting. The concepts of watermark removal and watermark forgery are quite practically relevant. Learned noise is injected to the watermarked images.

### Weaknesses
The title (and some parts of the abstract) does not align well with the content of this work.

The paper is poorly written and organized. In general, it is quite time-consuming to identify and fully understand the contribution of this work and the novelty of the proposed method. 

The attack method is based on the assumption that the underlying watermarks are embedded in the distortion layer between an encoder and a decoder of an image-generation process. It seems to me such applicability may not be necessarily broad. As cited by the authors, post-processing methods are also available, which are not tied to specific image-generation processes, such as diffusion models. It might be helpful to consider the effectiveness / robustness of the proposed method against watermarks embedded by post-processing methods.

Due to the assumption above, the proposed method hypothesizes that the extracted image features may exhibit certain clusters where some clusters correspond to information leakage. It is not clear to me if such an hypothesis could still hold when the underlying watermarking mechanism is changed.

Meanwhile, the proposed idea of identifying information leakage through extracted features was discussed in heuristic approaches in the existing literature, as mentioned by the authors. The novelty of the proposed method, in terms of generalizing this observation to deep learning frameworks with distortion layers, is not necessarily substantial.

### Questions
It seems to me that watermark removal and forgery correspond to type I and type II errors in a standard hypothesis testing setting (or false positive/negative, depending on what is defined as 1). Would the proposed method experience a tradeoff between the success rates of removal and forgery?

The ides of injecting noise to the image is intuitive. However, is it possible or effective to also inject noise at the embedding layer? Would a double-layer noise structure further improve the performance of the proposed method?

Does the proposed method require the unwatermarked images as the input, in addition to their watermarked counterparts?

### Soundness
3

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
This paper proposes a method for evading state-of-the-art watermarking approaches and also copying them. The insight is that many of the most robust watermark become more ``detectable'' slash learnable to some degree.

### Strengths
- The problem is interesting and relevant. It is interesting that we can do forgery and removal all in one paradigm. To my knowledge the method is new and novel.

- Empirical results show promise.

### Weaknesses
- The attack suite is not comprehensive enough. There are many attacks missing: rotation, regeneration, and others cited in popular benchmarks like WAVES.

- The number of methods evaluated is not enough and does not include true state of the art (older methods). Please include VINE [1] and Gaussian Shading [2], and etc. Note that this list is not exhaustive. I'm not convinced that the same pattern can be generalized to these settings.

- I noticed in the Appendix there was also discussion about the limitation to Tree-Ring. I think that weakness to this class of diffusion-based watermark that use inversion for embedding watermarks is significant in my opinion as this is a defining direction of the field. 

- Number of images is not enough (100).

- Algorithm may be slow as it needs to do it per image. (Thought it is parallelizable).


[1] Lu, Shilin, et al. "Robust watermarking using generative priors against image editing: From benchmarking to advances." arXiv preprint arXiv:2410.18775 (2024).
[3] Yang, Zijin, et al. "Gaussian shading: Provable performance-lossless image watermarking for diffusion models." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2024.

### Questions
- I'm not sure I understand this intuition given ``training watermarking models with increasingly diverse and severe distortions induces embedding redundancy''. Could you shed some more insight.

- I am still kind of confused by why outlier features/groups with fewer samples are examples of leakages.

- I'm wondering if you can somehow fully parameterize the optimization procedure to find the deltas and instead of doing it per-image create a general mapping that gives it to you automatically.

### Soundness
3

### Presentation
3

### Contribution
2

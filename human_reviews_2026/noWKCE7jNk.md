# PPGFlowECG: Latent Rectified Flow with Cross-Modal Encoding for PPG-Guided ECG Generation and Cardiovascular Disease Detection

- Avg Score: 4.00
- Decision: Withdrawn (Treated as Reject)
- Scores: 6, 2, 6, 2

## Abstract
In clinical practice, electrocardiography (ECG) remains the gold standard for cardiac monitoring, providing crucial insights for diagnosing a wide range of cardiovascular diseases (CVDs). However, its reliance on specialized equipment and trained personnel limits feasibility for continuous routine monitoring. Photoplethysmography (PPG) offers accessible, continuous monitoring but lacks definitive electrophysiological information, preventing conclusive diagnosis. Generative models present a promising approach to translate PPG into clinically valuable ECG signals, yet current methods face substantial challenges, including the misalignment of physiological semantics in generative models and the complexity of modeling in high-dimensional signals. To this end, we propose PPGFlowECG, a two-stage framework that aligns PPG and ECG in a shared latent space via the CardioAlign Encoder and employs latent rectified flow to generate ECGs with high fidelity and interpretability. To the best of our knowledge, this is the first study to experiment on MCMED, a newly released clinical-grade dataset comprising over 10 million paired PPG–ECG samples from more than 118,000 emergency department visits with expert-labeled cardiovascular disease annotations. Results demonstrate the effectiveness of our method for PPG-to-ECG translation and cardiovascular disease detection. Moreover, cardiologist-led evaluations confirm that the synthesized ECGs achieve high fidelity and improve diagnostic reliability, underscoring our method’s potential for real-world cardiovascular screening.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces a framework to synthesize difficult-to-acquire ECG from readily accessible PPG, which is a compelling and practically meaningful direction.

### Strengths
Proposing a model that is able to generate ECG from easy collecting PPG.

### Weaknesses
(1) The stated goal is to generate the ECG corresponding to a given PPG segment. The paper should first discuss whether, in realistic clinical settings, the ECG conditioned on a given PPG is uniquely determined.

If the mapping is deterministic:
[1] Using MAE/RMSE to quantify proximity between synthesized and reference ECGs is appropriate. However, within a deterministic framing, introducing stochastic noise at generation time leads to output variability, different noise produce different ECGs (EVEN WITH the same PPG input), which appears conceptually inconsistent. Why not train a direct PPG→ECG mapping without noise? I would expect MAE/RMSE to further improve under such a deterministic setup.
[2] Another issue is that the baseline generative methods compared in the paper are models that explicitly encourage high sample diversity. In my view, some of the proposed techniques (e.g., beat aligning) are in fact designed to reduce the generator’s “diversity” (which is considered an advantage in other works). Under this setting, the comparison seems not entirely fair. The paper should compare/discuss alternative ECG generation methods with less uncertainty (and even compare against a direct mapping model from PPG to ECG trained on the authors’ own architecture without diffusion). For example:
prior PPG→ECG works such as: Li Y, Tian X, Zhu Q, et al. Inferring ECG from PPG for Continuous Cardiac Monitoring Using Lightweight Neural Network. arXiv:2012.04949, 2020.
Or “Beats-align” related work: Biomedically Informed ECG Synthesis: Customizing Cardiac Cycle Phases with Diffusion Model, Y Lin, J Ma, W Wang, Z Wu, S Dong, G Luo, K Wang, 2024 IEEE BIBM.

If the mapping is non-deterministic:
[1] Then it is evident that using MAE/RMSE as the primary metrics may be unfair: although your method might produce ECGs closer to the label ECG, it would also indicate your method has poorer diversity compared with other models.

(2) Since the authors propose generating ECG from PPG, to demonstrate the benefit, readers would expect to see:
[1] Using PPG + generated ECG yields better diagnostic performance than using PPG alone;
[2] Using PPG + generated ECG achieves diagnostic performance comparable to using PPG + real ECG.

### Questions
The stated goal is to generate the ECG corresponding to a given PPG segment. The paper should first discuss whether, in realistic clinical settings, the ECG conditioned on a given PPG is uniquely determined.

### Soundness
2

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
5

### Summary
This paper presents PPGFlowECG, a two-stage generative framework for translating PPG into ECG signals for CVD detection tasks.  It introduces a CardioAlign Encoder and employs latent rectified flow to generate ECGs with high interpretability.

### Strengths
Introduce two-stage framework with shared latent alignment and latent flow generation on PPG-ECG domain.

### Weaknesses
The work has weak justification on the given task when it remains physiologically infeasible to use PPG for guiding ECG generation process due to their different nature.  

The motivation of using clinical data for translation is questionable.

The framework design as “Align first, Generate later” are not fundamentally novel. Also, the loss functions and ODE process are also introduced in prior works. 

Experimental results are not sound.

### Questions
1. Please discuss if PPG to ECG translation is theoretically reasonable and how to handle when the generated ECGs don’t contain wrong patterns (that might be serious in clinical setting). 

2. Related to the above point, a question raised is that do we really need the proposed translation framework with computationally cost complexity and latency (flow-based)? Please add an comparable experiment on using directly PPG encoder pretrained in Stage 1 (and/or other pretrained PPG encoder [1]) to finetune and evaluate on the downstream tasks, without relying on ECG or the generation process. 

3. Add another experiment without stage 1 to compare with current results (With stage 2 only, and unfrozen the Encoder and Decoder.). 

4. Related to the above point, in the proposed framework, since the ECG decoder in stage 2 is frozen, what is exactly the input of it in this stage? Is it different from the input of the ECG decoder in stage 1 (from 2 sourses: PPG and ECG)?  

5. Why the work only focus on single lead II generation, given that 12 leads generation is totally possible to conduct (is it because of the MCMED dataset only provide lead II and the authors are required to use this dataset ?) Also, all datasets used in this work are from clinical setting. However, in this setting, ECG is commonly accessible, plus the PPG and ECG here is much more stable (less sensitive about motion artifacts) than daily sensing data. So, the motivation of using clinical data for translation is questionable.

6. Line 459-461: adding more justification would be better why gen is better than real here? Also, why just use 50 signals (25 AF, 25 non-AF) in this experiment (which is too few for showing generalisation)? 

7. Please discuss the differences and provide possible comparisons with [2,3,4].  

8. The seen improvements when testing ablation studies are so minor, showing weak contributions of different loss functions (e.g., Table 3, 4).  

9. Provide additional experiments on different modeling approaches (e.g., diffusion-based) in stage 2, besides ODE. 

10. Section 4.5 seems to be overclaim. Figure 5 actually doesn’t show expected insights when the attention looks different and the signals also looks different between the original and generated ones. 

[1] Pillai, Arvind, et al. "Papagei: Open foundation models for optical physiological signals." The Thirteenth International Conference on Learning Representations (ICLR), 2025. 

[2] Li, Yuenan, et al. "Inferring electrocardiography from optical sensing using lightweight neural network." IEEE Transactions on Artificial Intelligence 5.7 (2024): 3535-3550. 

[3] Vo, Khuong, Mostafa El-Khamy, and Yoojin Choi. "PPG-to-ECG Signal Translation for Continuous Atrial Fibrillation Detection via Attention-based Deep State-Space Modeling." 2024 46th Annual International Conference of the IEEE Engineering in Medicine and Biology Society (EMBC). IEEE, 2024. 

[4] Nambu, Yuta, Masahiro Kohjima, and Ryuji Yamamoto. "CardioFlow: Learning to Generate ECG from PPG with Rectified Flow." ICASSP 2025-2025 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP). IEEE, 2025.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper proposes PPGFlowECG: a two-stage framework that first uses a shared CardioAlign encoder to align PPG/ECG to a unified latent space, and then performs conditional Rectified Flow on this latent space to generate ECG. Signal reconstruction and downstream CVD diagnostic evaluation are performed on multiple datasets such as MCMED, including clinical Turing Test and doctor interpretation experiments.

### Strengths
1. A clear "align first, generate later" design: shared encoder + distribution/instance alignment + cross-modal reconstruction, combined with rectified latent space flow, balances interpretability and sampling efficiency; outperforming GAN/diffusion/data domain flow baselines on multiple reconstruction metrics and downstream diagnostics.

2. Comprehensive evaluation: Four datasets (including clinical-grade MCMED, 10M+ aligned segments), unified preprocessing, providing MAE/RMSE/FD/FID/heart rate error and Macro-AUROC for multiple diseases, and performing ablation and external zero-shot validation.

3. Clinical usability signal: Turing Test is close to random (indistinguishable from real or fake), and replacing the real ECG with synthetic ECG can still improve/maintain the doctor's AF diagnosis F1 (0.94).

### Weaknesses
1. The boundaries of novelty need to be more clearly defined: Compared to the existing approach of "directly using Flow/diffusion in the data domain", this article's latent space Flow + multiple alignment is a combined innovation; it is recommended to make a more systematic comparison with "separate latent space generation or single alignment strategy" to strengthen the necessity argument.

2. Metric selection and consistency: In the external validation, signal-level metrics showed an "inconsistent metric criteria" phenomenon, with FD improving but FID/HR deteriorating. The authors' verbal explanation was reasonable, but they still recommended further analysis (such as correspondence with clinical factors).

3. Data protocol details: Except for MCMED, which uses the official split, other datasets use a subject-level 80/20 random split; it is recommended to disclose the number of subjects/window independence and repeated trial variance for each episode; currently only the main table with a single step size of T=10 is reported. The impact of the number of sampling steps is in the appendix, and a sentence can be added to the main text.

4. Clinical coverage: Only Lead II with a 10-s window is reported; generalization to rhythm-morphology complex lesions in multiple leads/longer time series remains unknown.

### Questions
1. Can the key tables be supplemented with multiple seed ± std/significance? Especially Table 1/Table 2 and the doctor reading experiment.

2. Deconstructing the sources of sliders for FD/FID/MAEHR in external validation (data domain differences, basic gradient distribution, noise spectrum)?

3. If I switch to multi-lead ECG or longer clips, what changes are needed in CardioAlign (does the shared encoder still hold true)?

4. If I switch to multi-lead ECG or longer clips, what changes are needed in CardioAlign (does the shared encoder still hold true)?

5. In this article, ECG is generated by PPG, which means it is essentially a process of information addition. Can you further explain why the additional information can improve the accuracy of the diagnostic results?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
PPGFlowECG is a two-stage translation framework: it first aligns PPG and ECG in a shared latent space via CardioAlign, then uses a rectified flow in latent space that deterministically transports PPG latents to ECG latents to reconstruct waveforms.
The model demonstrates effectiveness on multiple public, clinically oriented datasets using signal-quality metrics and downstream diagnostic tasks.
Clinician evaluations and real diagnostic tasks indicate that generated ECGs provide useful auxiliary value, suggesting utility approaching that of real ECGs.

### Strengths
- The architecture and training procedure are staged and clear, making the paper easy to follow.

- Data composition, preprocessing, and evaluation protocols are systematically organized, and implementation details are sufficient for high reproducibility.

- On downstream tasks, the method shows competitive performance for multi-label cardiovascular disease classification and AF detection, and ablation studies disentangle the contributions of each loss and the sampling configuration.

- Clinician assessments confirm perceptual fidelity and clinical support value of the generated ECGs, and Grad-CAM offers qualitative evidence that diagnostic attention regions are similar between real and generated ECGs.

### Weaknesses
- The core design couples cross-modal PPG to ECG translation with shared latent alignment and latent rectified flow, reading as a refined integration of existing alignment/generation frameworks. The authors position this as an align-first, generate-later contribution.

- Grad-CAM visualizations show alignment of diagnostic attention, but no dedicated quantitative interpretability metrics are reported.

- Explanations for why the latent flow improves physiological mapping rely more on intuition and empirical results than formal analysis.

- Comparisons against GAN/diffusion/flow baselines and extensive ablations are solid, but adding signal-level registration (e.g., DTW), VAE-based models, and linear latent alignment baselines would strengthen the case.

### Questions
It would be helpful if the authors could address this weakness, if possible.

### Soundness
2

### Presentation
3

### Contribution
2

# A Few Large Shifts: Layer-Inconsistency Based Minimal Overhead Adversarial Example Detection

- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 6, 6, 4

## Abstract
Deep neural networks (DNNs) are highly susceptible to adversarial examples—subtle, imperceptible perturbations that can lead to incorrect predictions. While detection-based defenses offer a practical alternative to adversarial training, many existing methods depend on external models, complex architectures, or adversarial data, limiting their efficiency and generalizability. We introduce a lightweight, plug-in detection framework that leverages internal layer-wise inconsistencies within the target model itself, requiring only benign data for calibration. Our approach is grounded in the **A Few Large Shifts Assumption**, which posits that adversarial perturbations induce large, localized violations of *layer-wise Lipschitz continuity* in a small subset of layers. Building on this, we propose two complementary strategies—**Recovery Testing (RT)** and **Logit-layer Testing (LT)**—to empirically measure these violations and expose internal disruptions caused by adversaries. Evaluated on CIFAR-10, CIFAR-100, and ImageNet under both standard and adaptive threat models, our method achieves state-of-the-art detection performance with negligible computational overhead. Furthermore, our system-level analysis provides a practical method for selecting a detection threshold with a formal lower-bound guarantee on accuracy.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper is on detection of adversarial inputs.
Evaluations of the proposed method are given on
CIFAR-10, CIFAR-100, and ImageNet datasets.

### Strengths
Detecting adversarial inputs remains a challenging problem.

### Weaknesses
The paper relegates to the appendix important issues such as the setting of the hyperparameters tau (line 156). Also, where does the "Lipschitz" parameters D come from in Assumption 1?  (For that matter, clearly define an adversarial input at this point.)

The fonts in Fig 1 and the ables are too small.  Within the
text, some of the mathematical expressions are hard to discern,
e.g., softmax on line 164 (which I don't think is the standard
definition).

L_RT involves a "benign" dataset x_n of N samples. How big does N need to be?
Considering the authors proposed regression of a "lightweight MLP", how
the number of benign samples should be a point of comparison
of different prior methods against the proposed ones, i.e.,
N should be clearly indicated in the tables for the different methods.
Are N samples employed per class in the proposed method?

The paper does not cite and compare against important related work, e.g., the early paper [1] (listed below) uses internal layers and compares the final layer activations against intermediate layers.  Other approaches build anomaly detectors during the training process, e.g., using (class conditional GANs [2] -- this method is easily applied to internal layers and the GANs could be learned post-training using enough benign data (N) as required train a lightweight MLP for regression.

[1] D.J. Miller et al.  Anomaly Detection of Attacks (ADA) on DNN Classifiers at Test Time.  Neural Computation 31(8), Aug. 2019.

[2] H. Wang et al.  Anomaly Detection of Test-Time Evasion Attacks Using Class-Conditional Generative Adversarial Networks.  Elsevier Computers & Security (COSE) 124, Jan 2023.

### Questions
See above.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
In this work, the authors present a defense against adversarial examples, leveraging two complementary strategies to perform detection. The first, Recovery Testing (RT), is based on training regressors to reconstruct intermediate features of each defended network's layer from its final layer's output. An input sample is labeled as adversarial if its inner layers' representations diverge from the reconstructed ones. The second, Logit-layer Testing (LT), applies input perturbations and computes features and logit discrepancies. The two approaches can also be combined in a single RLT score that (as RT and LT) can be thresholded to perform the detection.

### Strengths
- The paper is very clear and well-written.
- The authors justify each choice behind the design of their defense.
- The research problem is still open, and the provided contribution is relevant in this sense.
- The experimental evaluation shows a clear improvement with respect to the considered competing approaches.

### Weaknesses
- The provided robust accuracy under worst-case adaptive attackers reveals that the defense, in such a scenario (which is very relevant when considering security-related applications), is quite weak.

- (minor) Additionally, I'm a bit skeptical about adversarial example detectors, as they often have been broken by well-crafted attacks that are able to overcome the defense mechanism. The authors

### Questions
- As maximum-confidence adversarial attacks might overemphasize inconsistencies across layers, thus making them more detectable by the proposed defense, it would be very interesting to evaluate the defense against state-of-the-art minimum-distance attacks, such as FMN [a] or DDN [b]. Could you please provide some results about that?

[a] Pintor, M., Roli, F., Brendel, W., & Biggio, B. (2021). Fast Minimum-norm Adversarial Attacks through Adaptive Norm Constraints. ArXiv, abs/2102.12827.

[b] Rony, J., Hafemann, L.G., Oliveira, L., Ayed, I.B., Sabourin, R., & Granger, E. (2018). Decoupling Direction and Norm for Efficient Gradient-Based L2 Adversarial Attacks and Defenses. 2019 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 4317-4325.

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
This paper introduces the Recovery and Logit Testing Combined framework, a novel solution for adversarial example detection. It is founded on the "A Few Large Shifts Assumption." The method employs two complementary strategies to detect inconsistencies in the intermediate features and the final Logit layer, respectively. Operating as a lightweight, plug-in approach, this method enhances performance with less computational overhead.

### Strengths
1. The RLT framework it proposes is based on the A Few Large Shifts Assumption and includes a clear comparison with existing methods.  
2. The paper is clearly written, the figures and tables are easy to understand, and the overall flow of the text is good.  
3. The paper demonstrates advantages in both accuracy and efficiency compared to existing methods, and its effectiveness is validated through experiments.

### Weaknesses
1. The paper could benefit from further quantification of its core assumption. Although the phenomenon is empirically demonstrated through RT and LT, the work lacks a stronger theoretical or mathematical proof to explain why perturbations cause this local and disproportionate damage in "a small subset of layers" rather than being uniformly distributed across all layers.  
2. The applicability of the proposed framework, particularly the Recovery Testing (RT) module, is limited by its training data being restricted exclusively to benign samples. This leaves the discussion incomplete regarding its robustness on more diverse datasets.

### Questions
A Few Large Shifts Assumption is interesting. Are there any visualized experimental results available to validate this hypothesis? Furthermore, is there any specific analysis concerning these particular layers?

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
The paper proposes a plug-in detector that uses only the target model’s own internals and benign data to flag adversarial inputs. The core hypothesis, the A Few Large Shifts (FLS) Assumption, says adversarial perturbations create localized violations of layer-wise Lipschitz continuity at a small subset of layers. Two complementary tests are introduced: Recovery Testing (RT), which trains light regressors to reconstruct intermediate features from the last embedding and detects peaked reconstruction-error profiles; and Logit-layer Testing (LT), which learns a few mild, data-driven augmentations and detects cases where logits change disproportionately to internal feature drift. A fused score RLT combines both via quantile normalization. Experiments on CIFAR-10/100 and ImageNet report strong AUC under standard attacks and robust accuracy (RA) at fixed FPRs under adaptive attacks (Orthogonal-PGD; end-to-end PGD on the fused score). The paper also gives a system-level thresholding rule guaranteeing a lower bound on overall accuracy. (See Fig. 2 for the pipeline; Tables 2–3 for AUC; Tables 4–7 for adaptive/black-box; Fig. 3 for empirical support of FLS; Table 8 and App. G for overhead; App. H for system-level guarantees.)

### Strengths
1- Clear, self-contained detection paradigm. The method uses only the target network’s layer traces and benign calibration—no adversarial data, no external SSL encoder, no kNN graphs. The diagram on p. 3 (Fig. 2) cleanly shows RT/LT and their fusion.

2- Well-articulated mechanism + measurable proxies. The FLS assumption is made operational via layer-wise reconstruction errors (RT) and logit-vs-feature sensitivity ratios (LT), tied to local Lipschitz language and formalized in App. B with theorems motivating separability and the RLT fusion

3- Strong empirical coverage (CNNs and ViT). On CIFAR-10/100 and ImageNet, RLT matches or beats baselines—including BEYOND—with far lower overhead (see Table 2 on p. 6 and Table 3 on p. 6; Tables 17–18 extend to multiple CNNs and ViT-B/16).

4- Adaptive-attack evaluation beyond “paper rules.” The paper includes Orthogonal-PGD (Table 4), end-to-end PGD directly optimizing −Lcls + λ·RLT (Table 5), and a gradient-free SimBA stress test (Table 7) to address gradient obfuscation concerns.

5- Overhead is explicitly quantified and tunable. Table 8 (main) and Tables 19–22 (App. G) report FLOPs/params/size; the method attains high AUC with small MLP regressors and few learned augmentations (G≤4; App. D.2).

### Weaknesses
1- Threat-model precision and attack breadth. While Orthogonal-PGD, end-to-end PGD on RLT, AutoAttack, and SimBA are included, the main tables emphasize ℓ∞ (and one ℓ2 ablation). Missing: EOT against any stochasticity in the learned augmentations, multi-restart ablations for PGD (step-size/steps/restarts grids), and transfer from surrogate models in a detection sense. 

2- Assumptions around RT "invertibility." RT presumes the existence of approximate inverses from $z_L$ to earlier $z_k$ (Assumptions 2-3). This is plausible but not tested for distribution shift, different pre-act placements, or non-image domains; stability may be architecture- and dataset-dependent.

3- Learned augmentations (LT) could be attackable. The augmentations $W^{(g)}$ are learned to preserve benign logits yet change features. An adaptive attacker can coopt this structure (differentiate through BPDA) to reduce the LT ratio (App. B. 5 notes conflict but gives no worst-case rate). A stronger evaluation with EOT or randomized augmentation families would shore this up.

4- Quantile normalization as a potential leakage channel. RLT relies on benign-fit CDFs; an adaptive attacker with knowledge of this mapping (Perfect-Knowledge) could target score massaging near quantile knees. The paper implements BPDA through quantiles but does not include explicit attacks that optimize post-quantile loss landscapes.

5- Calibration vs. cost tradeoffs not fully charted. App. G shows pointwise overheads and some Pareto comparisons, but there is no global Pareto frontier of (AUC/RA@FPR, FLOPs, Params, latency) across kRT/kLT, MLP width/depth, G—useful for deployment decisions

6- Potential class-imbalance or per-class heterogeneity. Detection is reported as overall AUC/RA@FPR; per-class TPR/FPR (or per-image difficulty) aren’t shown. Since RT learns regressors over all benign data, hard classes might calibrate poorly.

7- ImageNet evaluation is partial. Table 3 omits AutoAttack for baselines “due to resources,” complicating SOTA claims; only DenseNet-121 is used as the ImageNet backbone; no results for BN-heavy architectures or modern ViT-L/ConvNeXt-XL on ImageNet.

8- Robustness to benign distribution shift. App. I tests random noise at fixed label (Table 27) which is helpful, but broader benign shifts (blur/lighting/cropping, CIFAR-C, ImageNet-C) and OOD (e.g., CIFAR-10.1/10.2) are not shown; false positives could rise on benign shifts.

### Questions
1- PGD tuning & multi-restart. Please report step-count/step-size/restart sweeps for PGD (e.g., steps 10→200, step-size ε/10→ε, restarts 1→50), and show RA@FPR at the strongest discovered settings—especially for the end-to-end PGD on RLT in Table 5.

2- EOT / stochasticity. Are the learned augmentations $W^{(g)}$ deterministic at test time? If not, add EOT-APGD and EOT-SimBA; if yes, test randomized ensembles of $W$ at inference and evaluate EOT against that randomized LT.

3- Transfer-based detection attacks. Beyond SimBA, can you craft surrogate-modelguided inputs that both fool $f$ and minimize RLT (e.g., via a surrogate with similar RT/LT regressors/augmentations) and report transfer success?

4- Invertibility stress tests. Show RT performance when the recovery MLPs are mis-specified (too shallow/too narrow), when you freeze them after partial training, and under benign shift (CIFAR-C/ImageNet-C). Does RT become over-sensitive (FPR↑) on blur/contrast shifts?

5- Where is RT read from? For architectures with pre-act vs post-act conventions, batch-norm placements, and skip connections, how do choices of $z_k$ extraction points affect separability? Provide a small layer-selection study.

6- Adapting to $W^{(g)}$. Evaluate a white-box attack that optimizes $-\mathrm{Lcls}+\lambda \cdot \mathrm{LT}$ with BPDA through quantiles and, crucially, adds regularization toward the benign quantiles of LT (post-CDF) to directly exploit the fusion step. Does RT rescue RLT in that setting?

7- Randomized augmentation families. Replace fixed $W^{(g)}$ with a distribution over $W$ (e.g., low-rank affine or small spatial transforms) learned on benign data; report RA@FPR with and without EOT. 

8- Quantile-attack ablation. Craft attacks minimizing $R T_{\text {norm }}$ or $L T_{\text {norm }}$ post-quantile and compare with minimizing the raw RT/LT. Provide per-example scatter of (raw score, normalized score) to illustrate attackable knees in the empirical CDF.


9- Pareto frontiers. For a fixed backbone, chart (AUC or RA@FPR, FLOPs, params, latency) across $k_{R T}, k_{L T}$, MLP sizes, and $G$. This helps practitioners select deployments under cost constraints.

10- Broader datasets and tasks. Add ImageNet AA baselines (even if subset) and at least one non-vision or higher-resolution setting; include physical re-photography for a small subset to assess LT’s real-world stability.

11- Benign distribution shift. Report FPR under CIFAR-C/ImageNet-C at thresholds tuned on clean validation (as in Table 27 but with semantic corruptions), and provide per-corruption results.

## Suggestions

1- Add EOT, PGD-grid sweeps, and a quantile-aware adaptive attack to harden the evaluation; include transfer-based detection attacks.

2- Provide per-class TPR/FPR and distribution-shift FPR (CIFAR-C/ImageNet-C).

3- Publish Pareto curves (accuracy vs cost) for RT/LT/RLT

### Soundness
2

### Presentation
3

### Contribution
2

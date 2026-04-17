# PACE: Pretrained Audio Continual Learning

- Decision: Accept (Poster)
- Scores: 6, 8, 6, 6

## Abstract
Audio is a fundamental modality for analyzing speech, music, and environmental sounds. While pretrained audio models have significantly advanced audio understanding, they remain fragile in real-world scenarios where data distributions evolve over time. In this work, we present the first systematic benchmark for audio continual learning (CL) with pretrained models (PTMs) and provide a comprehensive analysis of its unique challenges. Unlike in the vision domain where parameter-efficient fine-tuning (PEFT) has proven effective for CL, directly applying such strategies to audio leads to poor performance. This is due to a fundamental property of audio backbones: they emphasize low-level spectral details rather than structured semantics, resulting in severe upstream–downstream misalignment. Through extensive empirical analysis, we identify a promising technical route based on analytic classifiers with first-session adaptation (FSA), but also uncover two major limitations: representation saturation in coarse-grained scenarios and representation shifts in fine-grained scenarios. To address these challenges, we propose **PACE**, an innovative method that improves FSA via a regularized analytic classifier and introduces multi-session adaptation through adaptive subspace-orthogonal PEFT for better semantic alignment. Additionally, we design spectrogram-based boundary-aware perturbations to mitigate representation overlap and improve stability. Experiments across six diverse audio CL benchmarks demonstrate that PACE substantially outperforms state-of-the-art baselines, representing a significant step toward robust and scalable audio CL with PTMs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper investigates the challenges of continual learning with pretrained audio models. Through a systematic benchmark, the authors identify that a mismatch between pretraining objectives and downstream tasks leads to representation saturation in coarse-grained scenarios and severe representation shifts in fine-grained ones. To address this, they propose PACE, a framework that improves upon First-Session Adaptation (FSA) with three main contributions: (1) a restricted adaptation strategy using later-layer LoRA to prevent saturation , (2) an adaptive multi-session, subspace-orthogonal PEFT to enable stable learning across tasks , and (3) a boundary-aware perturbation regularizer to enhance class separability. Experiments across six diverse audio benchmarks demonstrate that PACE substantially outperforms existing state-of-the-art methods.

### Strengths
- Problem Formulation and Analysis: The paper excels in its methodical diagnosis of the problem. The identification of distinct failure modes for coarse-grained (representation saturation) and fine-grained (representation shift) tasks is a key insight that strongly motivates the design of the PACE framework.
- Novelty of the Proposed Method: PACE is a novel combination of several well-reasoned ideas. The adaptive multi-session subspace-orthogonal PEFT is a clever solution to balance plasticity and stability, and the later-layer LoRA adaptation is well-justified by both empirical results and intuition about hierarchical representations in neural networks.

### Weaknesses
- Clarity of Experimental Reporting: The paper's primary weakness lies in the fragmented and unclear presentation of its ablation studies, which makes it difficult to systematically assess the contribution of each component of the proposed PACE method. The validation results are scattered between the main text and the appendix, and even within the main paper, the connection between experiments and claims is not always explicit. For example, the caption for Table 3 is too general, forcing the reader to cross-reference the methodology to understand that it validates the "restricted head learning" strategy. A consolidated "Ablation Studies" section that clearly and methodically evaluates each of PACE's innovations would significantly strengthen the paper's clarity and persuasive impact.
- Hyperparameter Sensitivity: The proposed method introduces several new hyperparameters (e.g., the CKA threshold, the SVD threshold, the MSA stopping criterion, boundary misclassification threshold, ...). While the values are provided, the paper lacks a sensitivity analysis. It would be beneficial to understand how robust the method's performance is to variations in these hyperparameters.

### Questions
Could the authors clarify the design choices and associated trade-offs of the boundary-aware perturbation mechanism? The method's reliance on the previous model to identify boundary samples seems to imply a memory overhead from storing past models, which could be a significant drawback in long-sequence CL scenarios. Furthermore, time-frequency masking is typically employed as data augmentation to encourage invariance by pulling perturbed samples toward their class centroid. By contrast, the proposed method uses these perturbations to define a boundary to push features away from. Could the authors elaborate on the rationale for this specific design and discuss whether it potentially compromises the model's learned robustness to these common audio transformations?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper studies continual learning for audio tasks with pretrained models. It argues that directly using parameter-efficient fine-tuning methods from vision does not work well for audio, since audio backbones focus on low-level spectral cues while continual learning needs high-level semantic adaptation. The authors provide a benchmark and analysis showing two failure modes: representation saturation on coarse-grained tasks and representation shift on fine-grained tasks. They propose PACE, which improves first-session adaptation, applies LoRA only to later layers, uses a subspace-orthogonal strategy for multi-session updates, and adds a boundary-aware regularizer. Experiments across six audio continual learning benchmarks show consistent gains over prior work and a reduced gap to joint training.

### Strengths
1. AFAIK, this paper is the first to provide a deep dive into the unique challenges of audio CL with PTMs. The empirical analysis that distinguishes the difficulties in coarse-grained vs. fine-grained audio scenarios (Findings 1, 2, and 3 in Section 2) is a major strength and provides a clear motivation for the proposed method.
2. Their PACE framework is technically sound and its components directly address the problems identified.
3. Their empirical evaluation is thorough. The authors benchmark PACE on six diverse audio datasets, with different granularities and domains. The comparison against a wide range of state-of-the-art CL methods (Table 2) is thorough and convincingly demonstrates the superiority of the method.
4. The paper well written. The figures are effective at conveying their ideas. The structured presentation of findings makes the paper easy to follow and understand.

### Weaknesses
We appreciate the detailed analysis and strong empirical results achieved in this submission. To maximize the impact and clarity of the work, we suggest the authors address the following points:

1. The central claim of the paper is about addressing a "fundamental property of audio backbones" in continual learning. Despite this broad claim, all experiments are exclusively conducted using the EAT backbone, which is a spectrogram-based masked prediction model. Demonstrating that the observed challenges (representation shift) and the effectiveness of the PACE solutions hold true for a distinct audio PTM architecture (e.g., BEATs, AST) would significantly strengthen the paper's claims of generality.

2. No explanation on how $\rho_{layer}=0.94$, $\rho_{svd}=0.99$, and $N_{stop}=220$ were chosen. There were multiple continuous variables that were defined without a proper analysis of their impact to the method's performance.

3. The Multi-Session Adaptation (MSA) stage is essential for fine-grained performance. The authors adopt complex strategies, including LoRA Subtraction and subsequent SVD on the uncentered covariance matrix $X^{ucov}_t$, specifically to ensure "computational efficiency" against methods with "extensive storage overheads". However, to fully validate this optimization, a quantitative analysis detailing the increase in memory footprint or training time (e.g., time per session) of the full PACE pipeline compared to simpler, stable FSA baselines (such as RanPAC or ACL) is required.

### Questions
1. The stopping threshold for Multi-Session Adaptation (MSA) is determined when the cumulative number of seen samples exceeds $N_{stop} = 220$, which then dictates the final adaptation session $T_3$ (Table 7). $N_{stop}$ appears to be a fixed hyperparameter across all six datasets. Could the authors elaborate on the selection process for the specific value of $220$?  Was this value chosen based on a grid search across datasets?


2. In the improved First-Session Adaptation (Section 3.2), the paper proposes an asymmetric training scheme where the head's learning rate ($\eta_{head}$) is set significantly lower than the backbone's ($\eta_{bb}$). The authors note this is opposite to methods like LAE and SLCA, which suppress backbone updates to mitigate forgetting. Could you specifically elaborate on why this "opposite" approach is beneficial for audio PTMs? I assume it has to do with "compelling the backbone to absorb most gradient signals", which aligns with the finding that adaptation should be restricted to the deeper, semantic layers ($\ell \geq L_{tune}$). Does this suggest that the task-specific semantic information (required by downstream audio tasks) is encoded so poorly in EAT's deeper layers that significant, forced fine-tuning via a high $\eta_{bb}$ is mandatory for effective transfer, unlike in pretrained vision models where freezing often is enough?


3. The boundary-aware regularization component (Section 3.3) uses time-frequency masking ($Q(x_{i,t}, r_T, r_F)$) to generate perturbed samples near decision boundaries (approximating $B_t$). The loss term $\mathcal{L}_{reg}$ is shown to further enhance performance gains provided by MSA (Figs. 6(c) and 6(d)). Given that many audio augmentations exist (e.g., adding noise, pitch shifting), is the choice of time-frequency masking critical to effectively generating the required "boundary-prone samples" $B_t$? A discussion or small ablation comparing time-frequency masking against other standard augmentation techniques for defining $B_t$ would confirm that the choice of perturbation method is specifically aligned with the spectral nature of audio features.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes PACE, an audio continual learning system that combines multiple techniques, including a low learning rate for head learning, Later Layer LoRA for backbone finetuning, Multi-Session Adaptation for continual learning, and Boundary-Aware Perturbation for enlarging the inter-class margin. It shows good results on various benchmarks.

### Strengths
1. The experiment results are impressive.
2. Analysis of the challenges of audio continual learning is clear and understandable.
3. The illustrations in the paper are good.

### Weaknesses
1. The technical points are scattered:

The techniques presented in this paper are rather fragmented, jumping from one idea to another without a unified narrative. This makes it difficult for readers to clearly understand the motivation behind each technique.

2. Limited novelty of the proposed methods:

The idea of using a lower learning rate for training the head seems more like an empirical tuning strategy rather than a fundamentally new contribution, and the concept of Multi-Session Adaptation is not novel.

3. Insufficient experiments:

The ablation study is not comprehensive enough. For fine-grained datasets, the paper lacks ablation results for each proposed component, and the accuracy V.S. session curves (similar to Figure 2(c)) are also missing.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This work studies continual learning on audio pretrained encoders. The authors argue that continual learning techniques designed for vision do not transfer well to audio and propose **PACE**, a continual learning framework tailored for audio pretrained models. 

**Contributions**

1. **Improved first session adaptation.** Freeze early layers and adapt only deeper and more task sensitive layers before moving to analytic inference.  
2. **Adaptive subspace orthogonal PEFT for later sessions.** Used to reduce forgetting and follows LoRA subtraction and null space continual learning ideas, so this part is adapted rather than fully novel.  
3. **Spectrogram boundary aware perturbations.** Encourage intra class compactness and inter class margin enlargement.  
4. **Regularized analytic classifier without rehearsal.** Uses an analytic classifier that avoids storing historical data.

### Strengths
- Well written and easy to follow.  
- Comparisons are fair and support the claims.  
- Addresses area of audio continual learning.

### Weaknesses
- **Learning vs forgetting.** Different contributions such as  projection based update that protects past tasks can also limit learning on the current task. Please add an analysis that separates how much performance comes from stability and how much is lost in plasticity.  
- **Fine grained tasks are mostly human voices.** Current fine grained results are on speech or voice like data for example TIMIT and VocalSet. A non voice fine grained audio task such as environmental or musical instrument classification would make the claim stronger.  
- **Compute and overhead.** PACE introduces extra stages and training which add compute and time requirements compared to RanPAC.
- **Single dataset sessions.** All sessions are constructed from the same dataset. Please add a cross dataset or cross domain setting to show robustness.
- **Only one pretrained Encoder (EAT) is used.** The experiments use only one audio pretrained encoder (EAT). Adding music oriented models such as MERT and speech specific encoders would show that PACE is not tied to a single backbone.

### Questions
1. You showed that vision continual learning methods do not transfer well to audio. Do you expect PACE itself to transfer to vision or are parts of it truly audio specific?  
2. It was a bit hard to situate PACE among existing audio continual learning works. Is this mainly because there are very few strong audio continual learning baselines?

### Soundness
3

### Presentation
3

### Contribution
2

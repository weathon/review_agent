# FraIR: Fourier Recomposition Adapter for Image Restoration

- Avg Score: 3.33
- Decision: Reject
- Scores: 4, 2, 4

## Abstract
Restoring high-quality images from degraded inputs is a core challenge in computer vision, especially under diverse or compound distortions. While large-scale all-in-one models offer strong performance, they are computationally expensive and poorly generalize to unseen degradations. Parameter-Efficient Transfer Learning (PETL) provides a scalable alternative, but most methods operate in the spatial domain and struggle to adapt to frequency-sensitive artifacts like blur, noise, or compression. We propose \textbf{FraIR}, a Fourier-based Recomposition Adapter for image restoration that enables efficient and expressive adaptation in the spectral domain. FraIR applies a 1D Fourier Transform to decompose token features into frequency components, performs low-rank adaptation via spectral projections with learnable reweighting, and reconstructs the adapted signal using an inverse transform gated by task-specific modulation. Integrated as plug-and-play modules within Transformer layers, FraIR is reparameterizable for zero-latency inference and requires less than 0.5% additional parameters. Extensive experiments across denoising, deraining, super-resolution, and hybrid-degradation benchmarks show that FraIR outperforms prior PETL methods and matches or exceeds fully fine-tuned baselines demonstrating strong generalization with minimal cost. Unlike prior Fourier-based approaches that focus on generative modeling or static modulation, FraIR offers dynamic, degradation-aware recomposition in frequency space for efficient restoration.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents FraIR, a frequency-domain adapter module for parameter-efficient image restoration. The core idea is to perform adaptation in the Fourier space by decomposing hidden features using 1D FFT, applying low-rank projections with learnable spectral reweighting, and recomposing signals via inverse FFT with a gated residual connection. Integrated into Transformer backbones as plug-and-play modules, FraIR aims to provide efficient, degradation-aware adaptation with minimal parameter overhead. The authors evaluate their method across multiple restoration tasks including denoising, deraining, super-resolution, and hybrid degradations, and claim superior performance over several PETL baselines.

### Strengths
1. The paper proposes a new spectral-domain PETL design, which is relatively underexplored in the context of image restoration.
2. The method is lightweight and reparameterizable, which is beneficial for deployment.
3. Experiments are comprehensive in terms of tasks evaluated (denoising, deraining, SR, hybrid degradation).

### Weaknesses
1. The central claim — that frequency-domain adaptation provides superior efficacy over other adaptation domains — is not rigorously validated. While the paper compares FraIR to prior PETL methods, it does not isolate the benefit of frequency adaptation. Specifically, the performance gains could stem from differences in backbone architecture, parameter count, or training settings, rather than the frequency-domain design itself. No formal analysis is provided to explain why FraIR performs better — e.g., visualization of learned frequency masks is only briefly mentioned and not systematically analyzed.
2. Experimental comparisons lack fairness and control. In many cases, FraIR is evaluated using different backbone networks and training protocols than the competing methods, making it difficult to attribute improvements to the adapter design alone. This undermines the credibility of the claimed superiority. The paper should provide controlled head-to-head performance comparisons (rather than only runtime & effeciency) of FraIR against other adapter designs (e.g., LoRA, Adapter, PromptIR) under the same backbone, dataset, and training conditions. Such comparisons are essential to validate the benefit of the proposed frequency-domain recomposition.
3. AdaIR, which is one of the most relevant and recent PETL baselines also targeting restoration tasks, is only compared in Table 1 (denoising). It is notably absent from other key experiments such as deraining, hybrid degradation, and super-resolution. This selective comparison reduces the transparency and completeness of the evaluation.

### Questions
1. Why are AdaIR results only reported in Table 1? It is a strong and recent method for restoration tasks and should be compared consistently across all benchmarks.
2. Have the authors conducted experiments where FraIR and other PETL methods (e.g., LoRA, Adapter, SSF) are implemented on the same backbone (e.g., SwinIR or EDT), trained under the same protocol? If so, these should be clearly reported.
3. Can the authors provide further analysis or visualizations of the frequency gates or spectral projections in FraIR to support the claim of degradation-aware adaptation?
4. How does FraIR perform when applied to restoration tasks with less structured frequency distortions (e.g., low-light enhancement or spatial occlusions)? Would the frequency-domain approach still be beneficial?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces FraIR, a frequency-domain adapter for parameter-efficient image restoration. FraIR applies a 1D FFT to token representations, transforming them into the frequency domain, and then performs low-rank, learnable modulation. When integrated into existing algorithms, the proposed method outperforms state-of-the-art Parameter-Efficient Transfer Learning (PETL) baselines.

### Strengths
1. Unlike most existing methods, the proposed FraIR performs PETL in the frequency domain through a 1D FFT, integrating low-rank adaptation with frequency-domain processing.

2. Extensive experiments demonstrate the effectiveness of FraIR.

### Weaknesses
1. The overall writing quality needs substantial improvement. For example, the final sentence in the Contribution section is incomplete. In addition, several equations are not numbered. The first equation in Section 3.1 appears to be incorrect due to mismatched tensor shapes, and the variable $z_{in}$ in the second equation is not explained. Moreover, the meaning of FraIR in the equations presented in Section 3.3 is unclear. Finally, the textual descriptions in Section 3.2 do not correspond well with Figure 2.

2. The performance of the baseline models is not reported in the main result tables (e.g., Tables 1–3), making it difficult to assess the actual improvements achieved by the proposed method. In addition, the results for the spatial-domain version of the model (not the LoRA version, i.e., without performing 1D fft in the first equation of Section 3.2) are missing from the ablation studies. Furthermore, several representative frequency-based PEFT methods are not included for comparison, such as Ref. [1].

3. The comparison presented in Table 1 appears to be unfair. Specifically, the results of PromptIR and AdaIR are obtained by training on all-in-one datasets, whereas the proposed method is trained only on denoising datasets (as stated in Table 15). In addition, the performance of the proposed approach is not competitive, as shown in Table 10. Furthermore, the best results for LPIPS and DISTS under the RealSR setting are not highlighted.

Reference

[1] Parameter-Efficient Fine-Tuning with Discrete Fourier Transform, ICML'24.

### Questions
It is unclear how the authors conclude that `FraIR shows strong performance in real-world image restoration` in Section 5, given that the datasets used in the experiments are primarily synthetic and standard benchmarks.

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
This paper proposes a frequency-based adaptation technique to fine-tune generalist transformer models for image restoration. To accomplish this, the authors construct a linear function as follows:

1). Project the input sequence of 1D image tokens to frequency space along the channel axis

2). Learns matrices U and diagonal \Lambda to gate the frequency components: U\LambdaU^T where the dimension of \Lambda is smaller than the input dimension.

3). Perform an inverse FFT to go back to the feature domain, weight by a learned scalar G, and add back the input

The authors can then train the matrix U and V as adaptors for downstream tasks in image restoration. They benchmark on a variety of different tasks by fine-tuning the base backbone for denoising, deraining, super resolution, and a hybrid noise + blur baseline. The proposed method moderately outperforms all of the baselines in most settings.

### Strengths
1). The proposed method performs a wide variety of image restoration tasks (denoising, deraining, super resolution, and hybrid tasks) at the SOTA level 

2). There is no overhead at inference time and the method requires using a small amount of task-specific training data. 

3). The idea to leverage the frequency space to improve image restoration methods is nice and allows the modulation of the rank of the learned projection.

### Weaknesses
1). The technical difference and comparison with FouRA is the most important weakness for me. Please clarify the exact differences with FouRA. The authors use notation g and V in the last paragraph of the introduction to justify the difference, but this notation is not used anywhere else. As far as I can tell, g is the diagonal of \Lambda. The only difference with FouRA is thus a slight delta in the parametrization of the linear function after projecting to frequency space: A \alpha B -> U \Lamba U^T. The difference must be clarified. 

2). There seems to be only one direct comparison to FouRA and this is strangely only on CNN backbones. The authors should provide a comparison to FouRA with transformer backbones on their proposed tasks and clarify the differences in training the two models. 

3). The entirety of the methodological technical contribution is small and hard to follow (Sec 3.2).

4). Figure 2 naming convention does not match 3.2. Indeed, it doesn’t match the end of the introduction either.

5). I don’t think the “Reparametrization for inference” section is well written or necessary for main as it just states the previous section again and mentions the function is linear. It also ignores fusing before MHA.

### Questions
1). Is the learnable gate G input-independent and a singular scalar? 

2). Why isn’t there a Fourier projection (and unprojection) in line 217?

3). Why perform the FFT on the channel axis? Why not the 1D or 2D token axes? I see the ablation on 2D FFT in the supplement, but I am confused about the comparison because it seems the 2D FFT is along the two spatial axes vs the proposed 1D FFT on the channel axis. 

4). Why use different backbones for different tasks?

5). Why do the authors benchmark with LPIPS/DISTS for some tasks but not others?

6). What’s the difference between G and g?

7). Can you provide additional experiments and exposition to compare to FouRA?

### Soundness
2

### Presentation
2

### Contribution
2

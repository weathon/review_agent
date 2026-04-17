# Latent Diffusion Pretraining for Crystal Property Prediction

- Decision: Reject
- Scores: 2, 6, 6, 8

## Abstract
Fast and accurate prediction of crystal properties is a central challenge in new materials design. Graph Neural Networks have emerged as powerful tools for this task due to their ability to encode the local structural environment of atoms within a crystal. However, these models are data hungry, and in practice, labeled data for crystal properties are very scarce. Pretrain–finetuning strategies, particularly those based on diffusion models, have shown promise in addressing these limitations. In this work, we introduce a novel latent-diffusion based pretraining framework designed to mitigate the data scarcity issue. Our approach integrates a Variational Autoencoder (VAE) with a diffusion model during the pretraining stage. The VAE encoder maps 3D crystal structures into a smooth latent space, within which the diffusion process is applied. This latent diffusion pretraining enables the graph encoder to effectively capture structural and chemical semantics from large-scale unlabeled data, which can then be finetuned for specific property prediction tasks. Comprehensive experiments on popular DFT datasets for property prediction reveal that CrysLDNet significantly outperforms both training-from-scratch and pretrained baselines, with average improvements of 6.93% and 7.83% on JARVIS and MP over the second-best baseline. Additionally, the learned representations remain robust under sparse data conditions and are expressive enough to correct DFT errors when finetuned with limited experimental data.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper presents CrysLDNet, a generative pretraining approach that applies latent diffusion and denoising techniques to improve crystal property prediction. The authors aim to mitigate data sparsity and bias in chemical datasets by introducing a semi-supervised generative framework. The concept is well-motivated, as limited and unevenly distributed data are longstanding challenges in the field. The model integrates generative pretraining with downstream regression tasks and reports improvements across multiple benchmarks, suggesting enhanced generalization and robustness.

However, the experimental validation is limited by the choice of benchmarks and comparison baselines. The reported baselines are outdated and do not reflect the current state-of-the-art in crystal or molecular property prediction. For example, in formation energy prediction, recent models such as coGN (MAE = 0.0170) and DimeNet++ (MAE = 0.0235) outperform CrysLDNet by a noticeable margin. This raises concerns about the model’s competitive relevance. Moreover, the core architecture of CrysLDNet is relatively simple and appears to be a straightforward combination of existing denoising and autoencoding components, leaving little room for methodological novelty.

In addition, some claims in the manuscript require clarification. The authors refer to the model as a “Variational Autoencoder,” but the provided architecture and loss formulation resemble a standard autoencoder without explicit stochastic latent variables or KL regularization. Similarly, Figure 3 shows relatively low reconstruction accuracy (≈40%), which questions whether the latent representation sufficiently captures meaningful information. A correlation analysis between reconstruction fidelity and downstream prediction performance could strengthen the argument for the model’s effectiveness.

Overall, while CrysLDNet is simple and easy to follow, its contribution appears incremental, and its performance does not reach the level of more recent diffusion-based or graph-equivariant methods.

### Strengths
- Straightforward, interpretable model design.

- Demonstrates moderate improvement over the paper’s own baseline models.

- Addresses a relevant challenge in materials informatics — data sparsity and generalization.

### Weaknesses
- Reported benchmarks are outdated and exclude stronger recent baselines.

- Performance is not competitive with current state-of-the-art models.

- Architectural novelty is limited — appears as a simple combination of diffusion and autoencoder elements.

- Lack of clarity regarding the VAE structure and loss formulation.

- Insufficient analysis of reconstruction quality and computational efficiency.

### Questions
1. The model is described as a VAE, but there is no clear evidence of stochastic encoding or KL divergence losses. Could the authors clarify whether the model includes such components?

2. What specific denoising or diffusion steps are used in the latent pretraining? Are they based on DK losses or noise sampling as in typical diffusion models?

3. In Figure 3, reconstruction accuracy remains around 40%. How do the authors justify this as sufficient to capture key physical or structural features?

4. Could the authors provide a more detailed comparison to modern models such as coGN, ALIGNN, or Equiformer for a fair evaluation?

### Soundness
2

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes CrysLDNet, a Latent Diffusion pretraining framework for crystal property prediction. It addresses the limitations of existing diffusion models (like DPF) that operate directly on the non-smooth, high-dimensional input space. CrysLDNet is a two-stage process: 1) A VAE (with a Matformer encoder) compresses the crystal structure into a smooth latent space $Z$. 2) An LDM (DiT architecture) pretrains on this latent space $Z$ via denoising to optimize the encoder. The pretrained encoder $\mathcal{E}_{\phi}$ is then finetuned for downstream tasks

### Strengths
- Novel Framework: Moving diffusion from the high-D input space to a VAE latent space is well-motivated and addresses clear limitations of prior work .

- SOTA Performance: Comprehensively outperforms all baselines, including DPF and Matformer, on all 13 property tasks across JARVIS and MP (e.g., 6.93% & 7.83% average improvement).

- Data-Efficient: Demonstrates strong robustness and practical value in sparse-data regimes (Fig 4) and when correcting DFT errors with limited experimental data.

### Weaknesses
- Baseline Fairness: CrysLDNet uses the strong Matformer as its encoder. It is unclear if SOTA baselines (like DPF) use the same backbone, making the source of the performance gain (latent diffusion vs. better backbone) ambiguous.

- Pretraining Complexity: The two-stage (VAE + LDM) process is more complex than single-stage models like DPF; the associated training cost is not discussed.

- Unclear Ablation: The "LDM Only" ablation  is ambiguously described, making its distinction from the second stage of the full CrysLDNet pipeline unclear.

### Questions
- Baseline Encoder: What GNN encoder does the SOTA baseline DPF  use? How can you ensure the gains are from your latent diffusion framework and not just the Matformer backbone?

- "LDM Only" Ablation: Please clarify the "LDM Only" ablation setting. Did it start with a randomly initialized Matformer encoder?

### Soundness
3

### Presentation
3

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
This paper proposes a new pretraining/finetuning method for crystal property prediction. The goal of pretraining is obtaining a good feature encoder of crystal structures from an unlabeled dataset; for this the authors first train a standard VAE, and then further train the encoder along with a flow network using conditional flow matching loss based on optimal transport conditional vector field. At fine-tuning stage, the encoder is postfixed with a property prediction head and fine-tuned with a labeled dataset. On the standard benchmark setups including both synthetic and experimental data, as well as in scarce data regimes, the authors report an improved performance over state-of-the-art baselines.

### Strengths
S1. The proposed method is, while being a combination of existing components, technically sound as far as I can confirm (with one caveat in W1), and it empirically achieves strong performances in various settings compared to existing methods. The empirical evaluation setup is sound as far as I understand; the baselines include recent state-of-the-art methods, and the hyperparameters for fine-tuning are chosen consistently across datasets (Appendix C.1).

### Weaknesses
W1. In Section 4.2.2, the authors claim that the VAE encoder and the flow network (learned vector field) are trained jointly with the matching loss in Equation (5). Usually, this type of training can easily collapse due to the presence a local optima, where the VAE encoder can simply collapse to produce a constant output, because then it becomes trivial for the flow network to reach zero loss by predicting the direction towards the constant target. Proper discussion and/or analysis on why such phenomenon does not happen seem necessary.

W2. Can the authors discuss why/how latent flow matching improves the quality of representation from the VAE encoder? This is validated in Table 3 but its underlying reason is not immediate to me.

W3. This is a minor comment, but I believe it is necessary to cite the original flow matching paper because the noise schedule and regression objective function are borrowed from there. Although OT flow matching is dual to a Gaussian diffusion, regressing the vector field is essentially taking the flow matching and probability flow ODE viewpoint.

### Questions
I don't have particular questions but would like to hear the authors' opinion on the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The authors propose CrysLDNet, a novel pretrain-finetune framework for crystal property prediction designed to address the problem of labeled-data scarcity . The core idea is to move the pretraining task from the high-dimensional, heterogeneous input space (atom types, coordinates, lattice) to a more compact and "smooth" latent space.

### Strengths
1. It is novel to consider the latent diffusion as a pre-training task for a downstream property predictor. The paper correctly identifies a key weakness in existing diffusion pretraining models like DPF.

2. CrysLDNet achieves state-of-the-art results on all 13 property prediction tasks across both the JARVIS and Materials Project datasets

3. The paper is very well-written. The introduction clearly motivates the problem, the related work section precisely identifies the limitations of prior art, and the proposed method (Figure 2) is logical and easy to follow.

### Weaknesses
The entire focus of the paper is based on the claim that VAE creates a "smooth" and "compact" latent space $Z$, which is then "refined" and "enriched" by the LDM. It is unclear about the definition of smoothness or compactness. Without this in-depth analysis, the success of this model is still a black box.

### Questions
1. Can you please provide an analysis of the latent space learned by the encoder?

2. You claim the LDM "refines" and "enriches" the latent representations. Following up on Question 1, can you provide any evidence of this?

3. Should line 16 in Algorithm 1 be $Z^0 \sim \mathcal{N}(0, I)^{N \times d}$ to match the per-node loss in line 19?

### Soundness
3

### Presentation
3

### Contribution
3

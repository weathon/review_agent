# Two-Layer Convolutional Autoencoders Trained on Normal Data Provably Detect Unseen Anomalies

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
Anomaly detection refers to the techniques that identify (probably unseen) rare or suspicious data that deviate significantly from the pre-defined normal data (Chalapathy & Chawla, 2019; Ruff et al., 2021). Empirical studies have observed that generative models trained on normal data tend to produce larger reconstruction errors when reconstructing anomalies. Based on this observation, researchers have developed various anomaly detection methods, referred to as reconstruction-based anomaly detection (RBAD) (Lv et al., 2024; Li et al., 2024) in the literature.

Despite the empirical success of RBAD, the theoretical understanding of RBAD is still limited. This paper provides a theoretical analysis of RBAD. We analyze the training dynamics of a 2-layer convolutional autoencoder and introduce the cone set of the features. We prove that the cone sets of the normal features would absorb the (convolutional) kernels of the autoencoder during training and use these absorbed kernels to reconstruct the inputs. The absorbed kernels are more aligned with the normal features, which explains the cause of the reconstruction error gap between the normal data and the anomalies. Synthesized experiments are provided to validate our theoretical findings. We also visualize the training dynamics of the autoencoder on real-world data, demonstrating our proposed cone set intuition.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper addresses the insufficient theoretical understanding of reconstruction-based anomaly detection (RBAD) and proposes a new theoretical framework introducing the concept of a “cone set” to describe the dynamics of feature learning during the training of convolutional autoencoders (CAEs). Through rigorous derivations, the authors demonstrate how convolution kernel parameters are gradually attracted to the “cone set” and align with corresponding true feature directions. Overall, the theoretical contribution lies in establishing a formal analytical foundation for RBAD, filling a major gap in the literature where theoretical explanations for autoencoder-based anomaly detection were largely absent.

### Strengths
1. The paper’s greatest strength lies in constructing a systematic theoretical framework for RBAD. While previous works achieved strong empirical performance, they lacked a clear explanation for why reconstruction error distinguishes anomalies. Through the cone set and feature absorption mechanism, this paper is the first to explain—via gradient dynamics—why AEs tend to retain only normal features while poorly reconstructing anomalies. The discussion in VPDM [1] of the “identical shortcut” phenomenon indirectly supports this view: if a model relies excessively on raw input rather than stable feature extraction, it may reconstruct anomalies as well, leading to detection failure. The cone set theory precisely explains that models which genuinely learn features, rather than memorizing inputs, naturally exhibit poor reconstruction for anomalies.
2. The theoretical formulation is highly structured, with well-defined assumptions, lemmas, and proofs. The use of Hilbert space representations for patches, smooth ReLU activations for differentiability, and global max pooling to avoid gradient entanglement all demonstrate careful mathematical design. 
3. The paper’s theoretical results align with long-standing community observations. For instance, it formally explains why RBAD struggles with semantic anomalies (since they may contain normal local features) and why mild overparameterization helps avoid suboptimal reconstructions. These insights transform empirical heuristics into theoretically grounded knowledge, strengthening both credibility and interpretability.

[1] Li Y, Feng Y, Chen B, et al. Vague Prototype-Oriented Diffusion Model for Multi-Class Anomaly Detection

### Weaknesses
1. The theoretical analysis relies on several idealized assumptions. For example, the data are assumed to consist of P patches, each containing one dominant feature plus random noise. Real images are more complex, patches may not be independent and could contain multiple mixed features. Similarly, the definition of anomalies as “patch replacements” may not cover global distribution shifts or complex anomaly patterns. 
2. The paper only analyzes a two-layer convolutional autoencoder (single hidden convolutional layer + pooling). In practice, anomaly detection models are often deeper and structurally richer. The conclusions may not directly generalize to these more complex architectures, and care must be taken when extrapolating.
3. The main limitation lies in the narrow scope of empirical validation. Although the paper includes some experiments, they are limited compared to the complexity of anomaly detection tasks. The synthetic experiments, while rigorous, remain toy settings far from real-world conditions. 
4. The paper barely discusses or compares with current state-of-the-art (SOTA) anomaly detection methods. While pure theory need not compete in performance, the omission creates a sense of detachment: the theory explains a relatively basic method, while the field’s focus has partially shifted toward more advanced ones. This gap could reduce confidence in the theory’s practical relevance.

### Questions
1. Could the theoretical assumptions be relaxed, or could the authors discuss whether similar principles hold for other generative models (e.g., diffusion models, GANs)? Additionally, if multiple patches are replaced (non-local anomalies), would the conclusions still hold?
2. Given the theoretical weakness of RBAD on semantic anomalies, could auxiliary tasks or regularization be introduced to enforce lower reconstruction ability for non-primary features?
3. The experimental section should be expanded.
4. In the revised version, clarify the relationship between this paper’s theoretical focus and existing empirical advances—emphasize that the goal is to complement, not compete with, SOTA detection methods.

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a new anomaly detection method, which provides a theoretical analysis of reconstruction-based
anomaly detection (RBAD). It constructs a two-layer convolutional autoencoder to reconstruct data and proves reconstructing normal data is easier than anomalies.

### Strengths
- This paper proposes an autoencoder model to reconstruct the normal data and anomalies, which trains on normal data. And it shows that the reconstruction error of normal data is smaller than that of anomalies.
- Theoretical analysis support the observation and conclusion of this paper.
- Experimental results on synthesis dataset validate the theoretical findings.

### Weaknesses
- How to use the non-semantic anomaly and semantic anomaly during training phase.
-The organization of section 3 is a bit chaotic. It is unclear to me how this method works.
- I am confused with the experimental results on real data. I don't know how to observe Figure 5. Please provide more explanations. 
- Why not detect anomalies directly based on the real datasets. Then we can see quantitative results of Acc and F1.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper gives a theory for why reconstruction-based anomaly detection (RBAD) succeeds on normal data yet struggles on anomalies by analyzing a two-layer convolutional autoencoder with max pooling.
It introduces cone sets to show that training aligns kernels to frequent normal features.
This alignment yields weak activations and large reconstruction errors for non-semantic anomalies,
while yields small reconstruction errors for semantic anomalies containing learned features.
The proposed theory is validated on both synthetic and real datasets.

### Strengths
- Although reconstruction-error–based anomaly detection is widely used, there has been little theoretical analysis of why normal data are well reconstructed while anomaly data are not. This paper provides a theoretical explanation for this.
- The paper is well written and uses figures effectively to explain complex theory, making it very easy to follow.

### Weaknesses
Please refer to the Questions section for details.

### Questions
- While this theory analyzes autoencoders, I am curious what role it would play for variational autoencoders (VAEs).
As generative models, VAEs assign high likelihood to normal data and low likelihood to anomalies.
That is, they reconstruct normal data well and fail to reconstruct anomalies.
However, as noted in [1],
VAEs can sometimes assign higher likelihood to anomaly data than to normal data.
Since a VAE can be viewed as a regularized autoencoder,
I wonder whether this theory applies.
Could this theory help explain that phenomenon?
- Could this theory also be effective for more complex architectures, such as ResNet-based autoencoders?

[1] Nalisnick, Eric, et al. "Do deep generative models know what they don't know?." arXiv preprint arXiv:1810.09136 (2018).

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
3

### Summary
This paper investigates the behavior of the convolution kernels in reconstruction-based anomaly detection (RBAD). To derive theoretical results, this paper assumes that image data can be decomposed into normal/auxiliary patch features and the model is a two-layer CNN. Under these assumptions, this paper reveals that the kernels are absorbed into the cone set oriented toward the normal feature. The paper claims that it can explain the reconstruction error of anomaly data becomes large and verifies theoretical results through synthetic/real-world data.

### Strengths
1. This paper is well written except for the section of experiment.
2. This seems the first theoretical explanation for reconstruction-based anomaly detection.
3. The absorption into the cone-set appears to be an interesting and reasonable.
4. Several theoretical results might inspire researchers to propose a new anomaly detection method or to improve existing one.

### Weaknesses
1. Experiments need more explanation and discussions. Section 4 does have the explanation about Figure 4.

2. Empirical evaluations seem insufficient. Which theoretical results do Fig. 4 and Fig. 5 verify?
I suspect there some theoretical results might be verifiable. For example, since Lemma 3.1 indicates that the norm of kernel increases according to the training step, 
it can be verifiable by plotting the norm vs training steps. I think the cone set is also verifiable by investigating values of the inner product in Def.3.1.
If not, why is it difficult and how do Fig. 4 and Fig. 5 support theoretical results?

3. Since fully-connected layers are also commonly used for anomaly detection in AE, a comparison with them might be interesting. 
For a fully-connected layer, if the input vector contains a normal feature linearly independent of the noise component, intuitively one would expect the weight matrix to have singular vectors in the direction of the normal feature. This would result in singular values of zero for the anomaly feature, leading to large reconstruction errors.
Considering that convolution is a special case of a linear layer [a,b], a similar argument holds. Under such assumptions, it seems likely that the frequency components of an image would be formulated as the normal feature. The kernel visualization in Fig. 5 might also support such explanation as each kernel possesses a specific spectrum.
Does the proposed theory have any advantages or more reasonable insights over such approach?

[a] Tsuzuku, Y., and Sato, I.  "On the structural sensitivity of deep convolutional networks to the directions of fourier basis functions". CVPR2019

[b] Sedghi, H., and et al. "The singular values of convolutional layers". ICLR2018

4. The contribution can be a bit weak if theoretical findings can only explalin the reactions to semantic anomalies or the magnitude of reconstruction errors for anomalies.
This paper would be strengthened if results can suggest directions for reconstruction-based anomaly detection researches or methods for improvement.
Can theoretical results suggest such directions or methods?

### Questions
Could you read Weakness and answer the questions?

### Soundness
3

### Presentation
3

### Contribution
2

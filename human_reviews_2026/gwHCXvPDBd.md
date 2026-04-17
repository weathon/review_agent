# NoisePrints: Distortion-Free Watermarks for Authorship in Private Diffusion Models

- Decision: Accept (Poster)
- Scores: 6, 2, 6, 4

## Abstract
With the rapid adoption of diffusion models for visual content generation, proving authorship and protecting copyright have become critical. This challenge is particularly important when model owners keep their models private and may be unwilling or unable to handle authorship issues, making third-party verification essential. A natural solution is to embed watermarks for later verification. However, existing methods require access to model weights and rely on computationally heavy procedures, rendering them impractical and non-scalable. To address these challenges, we propose $\text{\emph{NoisePrints}}$, a lightweight watermarking scheme that utilizes the random seed used to initialize the diffusion process as a proof of authorship without modifying the generation process. Our key observation is that the initial noise derived from a seed is highly correlated with the generated visual content. By incorporating a hash function into the noise sampling process, we further ensure that recovering a valid seed from the content is infeasible. We also show that sampling an alternative seed that passes verification is infeasible, and demonstrate the robustness of our method under various manipulations. Finally, we show how to use cryptographic zero-knowledge proofs to prove ownership without revealing the seed. By keeping the seed secret, we increase the difficulty of watermark removal. In our experiments, we validate NoisePrints on multiple state-of-the-art diffusion models for images and videos, demonstrating efficient verification using only the seed and output, without requiring access to model weights.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper provides some insights on how the embedding of the generated image is related to the random seed, and thereby proposes a watermark verification method. The method only verifies the dependency on the embedding and the random seed and no weight information is needed.

### Strengths
This paper provides an interesting angle to analyze the relationship between the random seed and the generated image. Theoretical result is also provided for H0 (though I understand there is difficulty in analyzing the underlying distribution under H1).

### Weaknesses
(W1) Practicality of the scenario: The paper assumes a setting where the model provider is different from the model user, and the user seeks to protect their IP. However, there are some concerns: (1) If the model provider is untrustworthy, why would the user choose to use that model in the first place? (2) If the model provider is trustworthy, what advantages does this approach offer over existing white-box methods, especially considering potential robustness concerns raised in (W3)?

(W2) Theoretical justification: When $E(x)$ and $h(s)$ are independent, the derived distribution appears reasonable. However, the distribution is unclear when $E(x)$ and $h(s)$ are correlated, leaving a gap in the theoretical analysis.

(W3) Robustness and empirical evaluation: Based on the theoretical results, it seems that the test’s power strongly depends on $E(x)$. This suggests vulnerability to attacks that substantially alter $E(x)$. In the examples in the paper, the embeddings remain relatively close to the originals. For more aggressive attacks, such as redrawing the image, the test’s power could degrade significantly. While the algorithm is lightweight, it would be useful for the authors to provide a comparison that examines the trade-off between computational cost and performance under stronger attacks.

### Questions
Please address my concerns in (W1) and (W3). I understand that it would be difficult to derive theories for (W2), but is it possible to provide an empirical distribution (histogram) on the test statistics under H0 and H1 respectively?

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
4

### Summary
This paper addresses the copyright verification problem for private models and proposes a novel method named NoisePrints. The core idea is to leverage the inherent correlation between the initial Gaussian noise seed used during generation and the final output image, treating this correlation as a natural and verifiable watermark. The authors also innovatively incorporate zero-knowledge proofs, enabling ownership verification in this scenario without exposing the random seed. Experimental results demonstrate that NoisePrints can effectively resist both common image processing operations and specific forgery attack, exhibiting excellent robustness.

### Strengths
1. The paper presents a method with an elegantly simple design. Its requirement of only a public VAE for verification makes it highly efficient, resulting in significantly faster performance than existing approaches.
2. The approach eliminates the need for training and preserves output quality, offering a plug-and-play integration capability for diverse diffusion-based architectures.
3. The incorporation of a zero-knowledge proof circuit is a notable strength. It enables secure third-party verification without exposing the secret seed, effectively mitigating a key risk in ownership attestation.

### Weaknesses
1. The proposed NoisePrints is a single-bit watermarking method, which is primarily designed for ownership verification. In contrast, many existing methods are multi-bit, offering the extended capability of tracing the specific user responsible for the generation.
2. The proposed scheme faces the same practical deployment challenge as Gaussian Shading[1] (GS). To remain distortion-free, it relies on a randomly sampled seed for each generation, which necessitates securely logging and managing a vast database of `(x, s)` pairs. This requirement creates significant operational overhead and scalability concerns in real-world systems.
3. A logical inconsistency is identified regarding the applicability of the Dispute Protocol. According to Section 3.3, the protocol is activated exclusively when "two parties submit conflicting authorship claims that both pass the verification test." This precondition is inherently incompatible with the threat model of a geometric removal attack, as defined in Section 3.2. In such an attack, the adversary's success is measured by causing the legitimate claim `(x, s)`to fall below the verification threshold τ, thereby making the "both pass" condition unattainable. Consequently, the protocol, as currently formulated, offers no recourse for the rightful owner in this common adversarial scenario.
4. A significant tension exists between the stated threat model and the technical prerequisites of the proposed method. The introduction emphasizes the challenge of watermarking for private models where "model weights remain private and are never shared." However, the verification protocol requires the model provider to use a publicly available VAE encoder. This creates a dependency that contradicts the scenario of a completely self-contained, proprietary model, as a provider wishing to keep their entire pipeline private would be unable to use NoisePrints.
5. A critical security flaw exists in the naive protocol (without ZKP). The requirement for the content producer to expose the seed `s` by submitting the pair `(x, s)` makes the scheme vulnerable to forgery. An adversary who steals this pair can exploit the public VAE to create a perturbed image `x'` that is perceptually similar to `x` but lies outside the verification boundary for the legitimate owner. The adversary can then present the pair `(x', s)` and successfully claim ownership, all without requiring access to the private U-Net. This breaks the security model under a stolen seed scenario.
6. The claimed robustness against geometric attacks is conceptually problematic. The resilience is not an inherent property of the NoisePrint signal but is entirely dependent on the Dispute Protocol's ability to apply a corrective inverse transformation. This process merely reverses a specific, pre-defined manipulation (e.g., rotation, scaling) prior to verification. It does not demonstrate that the watermark itself can survive true geometric distortion, which typically causes irreversible, non-aligned spatial scrambling. The same logic could theoretically be applied to any basic image processing attack (e.g., contrast adjustment, blur) if an effective "inverse operation" could be found and applied. Therefore, the credit lies with the corrective pre-processing within the protocol, not with the fundamental robustness of the NoisePrints method.
7. The evaluation of the Zero-Knowledge Proof (ZKP) implementation is relatively preliminary.
8. The threat model for the "Watermark Injection" attack appears to lack practical motivation. The scenario in which an adversary creates a forged image that is *visually similar* to the original while also embedding an *identical* watermark seems contrived. In practice, an adversary seeking to claim ownership would more plausibly create a *different* image (e.g., a novel artistic creation) and falsely associate it with a forged seed, rather than meticulously replicating the original content with the same watermark. The authors should either provide a stronger justification for the considered injection attack scenario or redefine it to reflect a more realistic adversarial goal.

### Questions
1. **Regarding Weakness 1**: NoisePrints already assigns a unique seed to each user as an identity identifier. This foundation could be directly extended to construct a simple multi-bit scheme, for instance, by allocating a subset of seeds to represent specific user IDs. However, the authors have not evaluated NoisePrints from this perspective. To comprehensively benchmark against state-of-the-art methods like Stable Signature[2] and Gaussian Shading[1], it is necessary to evaluate its performance in terms of traceability accuracy within a multi-bit framework.
2. **Regarding Weakness 3**: The description of the Dispute Protocol's usage scenario should be reorganized to align with the experimental setup and resolve the logical contradiction when facing geometric attacks.
3. **Regarding Weakness 4**:  To resolve the contradiction in the threat model, I recommend removing the requirement for a public VAE. Instead of relying on model providers to reuse a public VAE—which conflicts with the scenario of fully proprietary models—the model owner could entrust their private VAE to the fully trusted verifier. This approach would better align with the stated principle that "the verifier is the only trusted party" and that "model weights remain private and are never shared," while still enabling the verification procedure.
4. **Regarding Weakness 5**:  I recommend that the authors augment the threat model with specific strategies to mitigate the risk of seed exposure in the naive protocol version.
5. **Regarding Weakness 6:** The paper attributes geometric robustness to the NoisePrints method itself. However, the described mechanism relies entirely on the Dispute Protocol's ability to apply an inverse transformation to "correct" the image before verification. Could you clarify how this approach demonstrates inherent robustness of the watermark signal, as opposed to being a general pre-processing strategy that could theoretically be applied to any watermarking scheme? Furthermore, does this mean that NoisePrints' geometric robustness is ultimately limited to attacks that are both invertible and whose inverse transformation is known and included in the public set 𝒢?
6. **Regarding Weakness 7:** The paper demonstrates the functional correctness of the ZKP implementation. However, its security guarantees are primarily cryptographic. A critical remaining question is: how does the *watermark robustness* fare when the image undergoes attacks *before* the ZKP-based verification is performed? Specifically, if an attacked image `x'` (e.g., after JPEG compression, blurring, or a geometric transformation) is submitted for ZKP verification, will the circuit still correctly output `1` (indicating a valid watermark) when provided with the legitimate seed `s`? We recommend that the authors conduct a simple but essential robustness evaluation for the ZKP scenario to confirm that the strong robustness demonstrated in the standard setting is preserved when verification occurs within the ZKP circuit.
7. **Regarding Weakness 8:** The authors should consider evaluating more practical forgery attacks, such as those proposed in [3].

8.  The paper does not mention a specific method for mapping bits $PRNG(h(s))$ to Gaussian noise $\epsilon(h(s))$. How is this step concretely implemented?
9. It is unclear from the experimental description whether a single VAE was used for all verification tasks, or whether the native VAE corresponding to each generative model was employed. Could the authors clarify this point?

[1] Yang, Zijin, et al. "Gaussian shading: Provable performance-lossless image watermarking for diffusion models." *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition*. 2024.

[2] Fernandez, Pierre, et al. "The stable signature: Rooting watermarks in latent diffusion models." *Proceedings of the IEEE/CVF International Conference on Computer Vision*. 2023.

[3] Müller, Andreas, et al. "Black-box forgery attacks on semantic watermarks for diffusion models." *Proceedings of the Computer Vision and Pattern Recognition Conference*. 2025.

### Soundness
3

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
3

### Summary
The paper tackles the challenge of proving authorship over visual AI-generated content, especially for cases where the diffusion model is private and traditional watermarking approaches (which need model weights or modify outputs) are impractical or inefficient. The authors propose NoisePrints, a lightweight, distortion-free watermarking scheme that utilizes the random seed used to initialize the diffusion process as a watermark proof, exploiting the strong correlation between this seed’s noise and the generated content. The verification only requires the seed and output, with no changes to the generation process and no access to model internals. NoisePrints is validated on multiple state-of-the-art diffusion models for images and videos, demonstrating efficient verification using only the seed and output, without requiring access to model weights.

### Strengths
**Originality:**

NoisePrints introduces a technically novel approach by using the stochastic seed of diffusion models as a watermark, enabling efficient and model-agnostic authorship verification. The scheme also integrates cryptographic techniques, such as zero-knowledge proofs, to assure privacy and security in third-party verification scenarios.

**Quality:**

The paper demonstrates strong methodological rigor, offering a comprehensive security analysis and empirical validation across diverse models and datasets. Experiments show the watermark’s high robustness to output manipulations and adversarial attacks, while significantly reducing computational overhead compared to inversion-based methods.

**Clarity:**

The writing is clear and logical. Core ideas, threat models, and algorithms are well explained, with protocols articulated stepwise, making the contributions accessible to both technical and broader audiences.

**Significance:**

By removing the need for model internals and ensuring distortion-free watermarking, NoisePrints directly addresses real-world needs for copyright and provenance management in private and proprietary diffusion models. Its privacy-preserving design and scalability make it highly significant for the trustworthy deployment of generative AI and digital content protection.

### Weaknesses
- In the introduction (from line 52), the discussion of the method does not clearly convey the technical challenges involved. As a result, readers may be left with the impression that the approach is straightforward to implement, which could undermine the perceived significance of the contribution. The authors should better articulate the complexities and nontrivial aspects of their method.
- Tables 1 and 2 lack visual highlights or markers for the best-performing methods, making it difficult for readers to quickly identify key results. Clear visual cues, such as bolding or color highlights, are recommended to enhance table readability and emphasize the main findings.
- Sections 3.3 and 3.4 are dense with technical details and formalism, posing accessibility challenges for readers without deep expertise in diffusion models or cryptography. These sections would benefit from additional intuitive diagrams and plain-language protocol summaries to broaden accessibility.
- Figure 1 suffers from inconsistent font usage and the inclusion of citation notes within module labels, which detracts from the professionalism and clarity of the visual. The authors should standardize fonts and reconsider the figure layout for improved visual coherence.
- Figures 2 and 3 are overly large and contain dense information, resulting in plot axes that are difficult to read. The authors should revise these figures to more logically group and present the data, possibly by dividing them into multiple panels and increasing the size and clarity of key elements.

### Questions
1. As mentioned in the limitation: The verification scheme relies on access to the public VAE used by the diffusion model. When the VAE is not public or is heavily modified, the approach may be less applicable. How the author plans to address such issues? 
2. Can the authors better articulate the complexities and nontrivial aspects of their method in terms of the task of this paper.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes NoisePrints, watermarking framework for diffusion models that uses the random seed of the generation process as a proof of authorship. By leveraging the strong correlation between the initial noise and the final output, the proposed method secures the generation process with a cryptographic hash and optional zero-knowledge proof, which enables verification without accessing diffusion model weights. Experiments on image and video diffusion models demonstrate that NoisePrints achieves robust and efficient authorship verification under common content manipulations.

### Strengths
1. The writing of the paper is clear and well-structured.
2. The proposed method is simple but effective, especially in terms of robustness against different types of attacking.
3. Experiments are conducted on multiple diffusion models (including both image and video generation) and against various types of attacks, demonstrating the generality of the method.

### Weaknesses
1. A main concern is the applicability of the method, in the scenario considered in this paper, the verification of the watermark relies on public structure VAE. However, in practice, it is possible that some diffusion models may update or fine-tune their VAEs across versions. It is not clear that under this circumstance, whether the proposed method is still effective. 
2. The motivation for the considered scenario requires stronger justification. In real-world applications, a more common concern is that model owners aim to trace who is responsible for the malicious or unauthorized use of their models, or that data owners wish to verify whether their data have been improperly used to train a model. In contrast, if a regular user simply generates an image using the model, it is unclear why they would need to prove authorship of the generated content, or why others might contest such authorship.

### Questions
1. If the watermarked images gone through semantic level modification, such as style transfer, can the watermarked detection accuracy still maintain?
2. According to some studies [1], large diffusion models exhibit partial memorization of training images. In such cases, different seeds may yield visually or latently similar outputs, breaking the one-to-one correspondence between the seed and the generated content. This could lead to higher false positive rates in NoisePrints verification, since unrelated seeds might still produce embeddings that correlate above the verification threshold. Can the author provide evaluation results of proposed method in such case?

[1] Memory triggers: Unveiling memorization in text-to-image generative models through word-level duplication
[2] Understanding (un) intended memorization in text-to-image generative models.
[3] Extracting training data from diffusion models.

### Soundness
2

### Presentation
3

### Contribution
2

# Fast, Secure, And High-Capacity Image Watermarking With Text Autoencoded Text Vectors

- Avg Score: 4.67
- Decision: Reject
- Scores: 4, 2, 8

## Abstract
Most image watermarking systems focus on robustness, capacity, and imperceptibility while treating the embedded payload as meaningless bits. This bit-centric view imposes a hard ceiling on capacity and prevents watermarks from carrying useful information. We propose LatentSeal, which reframes watermarking as semantic communication: a lightweight text autoencoder maps full-sentence messages into a compact 256-dimensional unit-norm latent vector, which is robustly embedded by a finetuned watermark model and secured through a secret, invertible rotation.
The resulting system hides full-sentence messages, decodes in real time, and survives valuemetric and geometric attacks. It surpasses prior state of the art in BLEU-4 and Exact Match on several benchmarks, while breaking through the long-standing 256-bit empirical payload ceiling. It also introduces a statistically calibrated score that yields a ROC AUC score of 0.97-0.99, and practical operating points for deployment.
By shifting from bit payloads to semantic latent vectors, LatentSeal enables watermarking that is not only robust and high-capacity, but also secure and interpretable, providing a concrete path toward provenance, tamper explanation, and trustworthy AI governance. Models, training and inference code, and data splits will be available upon publication (code available in a the supplementary zip file).

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper presents a novel image watermarking method called LatentSeal, which shifts the watermarking paradigm from the traditional "bitstream" approach to a "semantic communication" model. It introduces a key-based latent-space rotation encryption mechanism, achieving a unified balance of high capacity, robustness, security, and semantic interpretability.

### Strengths
1. The method embeds meaningful textual information within the watermark, enhancing its interpretability.
2. The effectiveness and robustness of the watermarking scheme have been validated across multiple datasets and attack scenarios.

### Weaknesses
1. The structure of the proposed model is relatively simple and seems to be a combination of existing works.
2. The experimental metrics only consider BLEU-4 and EM, without incorporating measures related to watermark strength.

### Questions
1. The text encoder and watermark embedding model used in this work are based on existing approaches. Where does the core innovation of this paper lie?
2. The watermark model and text autoencoder in this work require staged training. Could an end-to-end training approach be considered instead?
3. The paper claims to "break the 256-bit payload limit," but this is achieved by compressing the text into a continuous vector space, which seems to differ from the traditional definition of "bits" in information theory. Could you clarify this?

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
This paper proposes LatentSeal, a watermarking method that embeds a message sentence into an image. LatentSeal encodes a message into a unit vector, applies a secret rotation, and embeds it into the image. The message is later decoded by extracting the embedded vector. Building upon VideoSeal, the embedder of LatentSeal is empirically shown to be robust against common image edits.

### Strengths
1. The motivation and overall method are clearly presented and easy to follow.

2. Using semantically meaningful vectors as watermarks is a natural and appealing idea.

### Weaknesses
This paper proposes LatentSeal, a watermarking method that embeds a message sentence into the image. LatentSeal encodes a message into a unit vector, applies a secret rotation, and embeds it to an image. The message is decoded after extracting the embedded vector. Building upon VideoSeal, the embedder of LatentSeal is empirically shown to be robust to common image edits. 



**Strengths:**

1. The motivation and main method are clearly explained and easy to follow. 
2. Using sematic meaningful vectors as watermark seems to be a natural and appealing idea. 




**Concerns:**

1. Comparison to existing work. My understanding is that LatentSeal is autoencoder + VideoSeal. It is helpful to further explain the difference and innovation compared to VideoSeal. 
2. Watermark detection. The detection mechanism and its justification remain unclear. Specifically, in Figure 1, $\hat{y}$ denotes the latent vector extracted from an image, and $\hat{y}{\mathrm{rec}} = E \odot D(\hat{y})$, where $E$ and $D$ represent the encoder and decoder, respectively. The authors claim that an image is authentic if $\hat{y}$ is close to $\hat{y}{\mathrm{rec}}$. However, for any in-distribution latent vector $y$, a well-trained autoencoder typically satisfies $E \odot D(y) \approx y$. This suggests that closeness between $\hat{y}$ and $\hat{y}_{\mathrm{rec}}$ alone may not imply authenticity, as the relationship between $\hat{y}$ and the true latent vector $y$ remains unknown. Could the authors clarify how the confidence score in Equation (4) effectively captures this relationship?
3. Secret Rotation. What is the motivation for introducing the secret rotation layer? Please provide an example or real-world scenario where this component is necessary or advantageous for security or robustness.
4. How does the method’s performance change if the latent vector dimensionality is altered (e.g., not fixed at 255)? Some empirical evidence would strengthen the claims of generality. 
5. I would suggest add more basline methods such as DwtDct [1], Stable Signature [2], RAW [3], and etc. Comparing with them in terms of the AUC-ROC score helps to understand the cost of embedding meaningful messages. 

[1] Cox I J, Miller M L, Bloom J A, et al. Digital watermarking[J]. Morgan Kaufmann Publishers, 2008, 54: 56-59.

[2] Fernandez P, Couairon G, Jégou H, et al. The stable signature: Rooting watermarks in latent diffusion models[C]//Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023: 22466-22477.

[3] Xian X, Wang G, Bi X, et al. Raw: A robust and agile plug-and-play watermark framework for ai-generated images with provable guarantees[J]. Advances in Neural Information Processing Systems, 2024, 37: 132077-132105.

### Questions
Please see Weaknesses above.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper proposes an image watermarking framework called LatentSeal that, instead of embedding arbitrary bits, embeds a 256D vector, which represents full-sentence textual messages (actually a set of tokens, e.g., 30). The mapping is performed with a text auto-encoder, where the encoder is derived from ModernBERT.  
The latent vector is robustly embedded into the image using a finetuned watermarking model adapted from VideoSeal. 
To ensure secrecy, the latent vector is secured through a secret, invertible rotation conditioned by a key, meaning only authorized decoders can correctly reverse the rotation and recover the message

The paper claims that the system is fast, secure, and offer a higher capacity than traditional bit-centric watermarking methods, while maintaining robustness against image attacks.

### Strengths
The problem related to data protection in the era of foundation models is quite important, and watermarking is certainly a key stream of research to that end. 

The paper is well written and the contribution seems reasonable, to the best of my judgement (I have not followed closely the literature on this topic recently). 

The lightweight design of the decoder should enable fast, real-time decoding, making it practical for real-world deployment

The code is actually provided in the supplemental material. 1

### Weaknesses
The paper explicitly states a major limitation: the robustness of the system against powerful, image-editing models is not addressed. The authors acknowledge that these models may be able to strip their watermark.

Minor: The font is two small in Figure 2.

### Questions
What happens if you encode random ids with your text auto-encoder?

### Soundness
3

### Presentation
3

### Contribution
3

# THE SELF-RE-WATERMARKING TRAP: FROM EXPLOIT TO RESILIENCE

- Decision: Accept (Poster)
- Scores: 8, 4, 6

## Abstract
Watermarking has been widely used for copyright protection of digital images. Deep learning-based (DL) watermarking systems have recently emerged as more effective than traditional methods, offering improved fidelity and resilience against attacks. Among the various threats to DL watermarking systems, self-re-watermarking attacks represent a critical and underexplored challenge. In such attacks, the same encoder is maliciously reused to embed a new message into an already watermarked image. This process effectively prevents the original decoder from retrieving the original watermark without introducing perceptual artifacts. In this work, we make two key contributions. First, we introduce the self-re-watermarking threat model as a novel attack vector and demonstrate that existing state-of-the-art watermarking methods consistently fail under such attacks. Second, we develop a self-aware watermarking framework to defend against this threat. Our key insight for mitigating this risk is to limit the sensitivity of the watermarking models to the inputs, thereby resisting re-embedding of new watermarks. To achieve this, we propose a self-aware deep watermarking framework that extends Lipschitz constraints to the watermarking process, regulating encoder–decoder sensitivity in a principled manner. In addition, the framework incorporates re-watermarking adversarial training, which further constrains sensitivity to distortions arising from re-embedding. The proposed method provides theoretical bounds on message recoverability under malicious encoder based re-watermarking and demonstrates strong empirical robustness against diverse scenarios of re-watermarking attempts. Moreover, it maintains high visual fidelity and demonstrates competitive robustness against common image processing distortions compared to state-of-the-art watermarking methods. This work establishes a robust defense against both standard distortions and self-re-watermarking attacks. Code available at https://github.com/SVithurabiman/SRW.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces the self-re-watermarking threat model, where an attacker can reuse the watermark encoder to embed a second watermark into an already watermarked image. The authors demonstrate that existing Encoder-Decoder-based watermarking methods are not robust against this type of attack and often recover the second watermark instead of the original one. To solve this problem, the paper presents a methodology that combines Lipschitz continuity and adversarial training to limit the effect of re-watermarking attacks. They evaluate this approach against both a simple self-re-watermarking attack and a gradient-based re-watermarking attack.

### Strengths
1. Novel threat model and contributions. It is akin to performing both a removal attack on the existing watermark and a forgery to add a new watermark.
2. Empirically and theoretically motivated.
3. Good results.

### Weaknesses
1. I did not find major weaknesses with the paper; using a second dataset for the experiments to show that the phenomenon is not dataset-dependent could be valuable.
2. The notation should be included in the main paper rather than the appendix. The paper should be readable without the appendix, which is not the case without the notation.
3. The writing could be improved. There is some redundancy in the writing that, if removed, could free enough space to include the notation in the paper. I provide more detail in the suggestions.

### Questions
Questions:
1. I am struggling to rationalize how, with your method, the decoder is sensitive enough to embed a watermark, but not sensitive enough to re-watermark an already watermarked image. Did you look at the delta-K plot when comparing x and x_w? For this to work, it would need most points below the y = x line. However, if so, what makes watermarking the first time and re-watermarking so intrinsically different?
2. In Section 5.4, you claim to evaluate the worst-case scenario, but use the local K bound rather than the global bound. Did you look at the global bound, and if so, did you find any meaningful difference?
3. Did you study re-watermarking with a different watermarking scheme than the original watermark? 
4. This last question is both a question and a suggestion. Did you study how your methodology affects the performance of removal and forgery attacks? Intuitively, your GBA attack behaves similarly to an adaptive removal attack if one sets the target to be random bits, similar to [1]. 

Suggestions:
1. The training objective and pipeline could be combined and shortened, with the lengthy description of the loss replaced with an equation. Also, Section 4.2 does not convey any meaningful information beyond what was already said earlier and could be removed.
2. I think having Sections 3.3 and 3.4 come before Sections 3.1 and 3.2 would improve readability and clarity, making the paper feel less redundant.

Typos:
1. Line 89, no colon after follows and a trailing period after the bullet point.
2. The word “based” in line 252 does not make sense to me.
3. Line 442-443, there is a space missing after the period at the end of the sentence.

[1] Lukas, Nils, et al. "Leveraging optimization for adaptive attacks on image watermarks." arXiv preprint arXiv:2309.16952 (2023).

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces a threat model called self-re-watermarking, where an attacker reuses the same encoder that embeds a new message into an already watermarked image. This can hijack ownership by making the decoder recover the attacker’s message.
Experiments demonstrate that existing deep watermarking systems fail under such an attack. They propose a defense solution with Lipschitz constraints to preserve the original watermark.

### Strengths
1. The paper is well-structured, with a clear progression. 
2. They apply Lipschitz constraints to watermarking and directly training against self-overwriting is reasonable and practically motivated.
Using a composite loss that explicitly accounts for overwrite scenarios is aligned with the threat model.
3. The paper compares against several well-known watermarking methods (HiDDeN, MBRS, etc).

### Weaknesses
1. Cross-model overwriting is more frequent than self-re watermarking for attackers to do in many real-world settings. Protectors , leaking the encoder is not accepted in practical applications. Protectors usually do not use the encoder after it leaks. The motivation or task setting is not reasonable. 
2. The defense mechanisms (Lipschitz awareness) are known techniques for bounding sensitivity; the novelty lies more in applying them to watermarking. This is fine but it means the new contribution is primarily the threat framing and integration, not a fundamentally new robustness technique.
3. All experiments use 128×128 images and 30-bit payloads. Many watermarking applications use higher resolutions (e.g., 256–1024) and variable payload sizes (32–100+ bits). 
4. The boxplot can be refined for better visibility, such as setting different intervals for each sub-figure in Fig. 1.

### Questions
Can you add the framework and visualization results into the main paper for better readability?

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
3

### Summary
This paper introduce a novel deep watermarking framework to defend against this self-re-watermarking attacks.  This attack involves the use of the same encoder to embed a different message into an image that has already been watermarked, which renders the original watermark irretrievable.  Their approach combines Lipschitz constraints on the encoder-decoder sensitivity and re-watermarking adversarial training to limit model sensitivity to input distortions caused by re-embedding. Empirical results demonstrate the new framework’s robustness against self-re-watermarking and various image distortions while maintaining high visual fidelity.

### Strengths
1. This paper proposes an innovative solution to address self-re-watermarking attacks. The integration of Lipschitz constraints and adversarial training provides a principled solution, supported by theoretical guarantees. 
2. The experiments are thorough and well-designed. they evaluate performance across multiple SOTA baselines, diverse attack scenarios  and standard image distortions. The results consistently show superior robustness while maintaining high visual fidelity.

### Weaknesses
1. Self-re-watermarking attacks appear to be resistible through engineered approaches. For instance, one could first verify whether an image has already been embeded watermark (based on decoder perplexity or by training a classifier). If watermarked, the watermark would not be re-embedded during subsequent attempts. Therefore, what advantages does this method offer compared to such approaches?
2. The paper does not analyze composite distortion attacks, such as performing various noise attacks followed by self-re-watermarking.
3. Appendix G demonstrates that re-watermarking results in a significant deterioration of image visual quality, yet the paper does not appear to analyze why this occurs. Furthermore, based on the cases presented in Appendix G, the new artifacts introduced after re-watermarking seem largely similar. Does this indicate limitations in the method?

### Questions
Please refer to weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

# SpeeCheck: Self-Contained Speech Integrity Verification via Embedded Acoustic Fingerprints

- Avg Score: 5.00
- Decision: Reject
- Scores: 2, 8, 6, 4

## Abstract
Advances in audio editing have made public speeches increasingly vulnerable to malicious tampering, raising concerns for social trust. Existing speech tampering detection methods remain insufficient: they often rely on external references or fail to balance sensitivity to attacks with robustness against benign operations like compression. To tackle these challenges, we propose SpeeCheck, the first self-contained speech integrity verification framework. SpeeCheck can (i) effectively detect tampering attacks, (ii) remain robust under benign operations, and (iii) enable direct verification without external references. Our approach begins with utilizing multiscale feature extraction to capture speech features across different temporal resolutions. Then, it employs contrastive learning to generate fingerprints that can detect modifications at varying granularities. These fingerprints are designed to be robust to benign operations, but exhibit significant changes when malicious tampering occurs. To enable self-contained verification, these fingerprints are embedded into the audio itself via a watermark. Finally, during verification, SpeeCheck retrieves the fingerprint from the audio and checks it with the embedded watermark to assess integrity. Extensive experiments demonstrate that SpeeCheck reliably detects tampering while maintaining robustness against common benign operations. Real-world evaluations further confirm its effectiveness in verifying speech integrity. The code and demo are available at https://speecheck.github.io/SpeeCheck/.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The submission proposes a mechanism for detecting manipulated voice speech.
First, a feature extraction is trained via constrative learning to be discriminative:
- robust to bening audio eidting
- fragile to malicious audio editing.

Second, this feature is embedded in the audio stream thanks to a watermarking technique.

At detection time, one compares the features decoded from the watermark with the feature extracted from the audio.
A large difference means that the audio has been maliciously modified.

### Strengths
A nice assemblage of different technology bricks.

### Weaknesses
### W0 Lack of bibliographic references

Embedding a hash/perceptual hash within the content via a watermarking technique is a very old idea. It pertains to the so-called semi-fragile watermarking schemes dedicated to content authentication. See, for instance, the references in the overview *Review on Semi-Fragile Watermarking Algorithms for Content Authentication of Digital Images*, Yu et al., MDPI Future Internet.

As a consequence, I challenge the first two contributions. Accessing the original speech recordings has never been considered as an option in the literature. Using watermarking to hide a signature/fingerprint/perceptual hash/feature (whatever the name) is not new. 
It appears to me that the primary contribution is the multiscale features sensitive to malicious manipulations. 


### W1 Remaining issues

Moreover, this literature usually elaborates on three problems:
- Watermarking should not destroy the perceptual hash. No consideration of this topic is given here. More broadly, it would be good to discover the reason behind false negatives: is it due to a watermarking decoding failure or due to the feature extraction failure?
Localization: This literature not only decides whether the content is authentic but also spots where the manipulation happened within some granularity. There is no localization in this submission.
- Security. The audio watermarking technique AudioSeal is on the shelves, The proposed feature extraction might be as well. Therefore, there is no secret key. An attack can manipulate the content, recalculate the new feature and re-embed it in the audio stream. No consideration about the security in the submission.

### W2 Benchmark

- Since the feature extraction is the main contribution, a dedicated benchmark is welcome.
- As for authentication, AudioSeal can hide a 16-bit message, but its decoder also gives a detection score per sample. The absence of watermark signifies the manipulation. A comparison with AudioSeal is definitely missing. (However, I assume here that one can learn a private AudioSeal)

### W3 Naive argumentation

The motivation summarized in Figure 2 is naive. No professional would ever consider SHA256 hash as a suitable feature representation for multimedia content. In the same way, Table 4 is obvious: deepfakes are not watermarked.

### W4 Experimental results
I have some concerns about the evaluation metric (paragraph Line 337). Note that the threshold is fixed.
- How can you compute an AUC if the threshold is fixed? Anyway, AUC is a really bad metric for applications requiring small FPR.
- How can you compute an EER if the threshold is fixed?
- Why is the FPR varying in Table 1? Table 1 deals only with positive (ie. benign) content. Therefore, there is no way to measure a FPR as this requires negative (ie. manipulated) content. Moreover, the fact that the FPR varies from one benign operation to another can not be correct. The same happens with Table 2 where FNR evaluations are wrong. Also, note that, in principle,  FNR + TPR = 100 and FPR + TNR = 100, which is not the case here. Conclusion: Table 1 should report only TPR and Table 2 only FPR.

### Questions
### Q1

Since most of the operations deal with the semantics, why not hiding the transcript Speech-to-Text inside the audio stream with the help of LLM zip. Like this paper did for images: *FAST, SECURE, AND HIGH-CAPACITY IMAGE WATERMARKING WITH AUTOENCODED TEXT VECTORS*, Evennou et al.

### Q2

Why the TNR is lower than 100 for the voice conversion operation? As for Text-to-Speech. Isn't it a new audio stream not-watermarked?

### Q3

Embedding rate. The 256-dimension feature is embedded into 16 segments. How many seconds does this represent? What happen next if the audio stream is longer? Do you extract and embed another feature or is it always the same "global" feature which is embedded repeatedly?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper proposes SpeeCheck, a proactive speech integrity verification design, which is sensitive to tampering attacks, robust to benign operations, and self-contained. SpeeCheck is based on several delicate design including multi-scale feature extraction + contrastive learning, and the key observation that the fingerprint of a watermarked audio is almost the same as the fingerprint of the original audio.

### Strengths
- The paper accurately points out the insufficiency of the current detection methods.
- The paper proposes a solid solution. The choice of multi-scale feature extraction + contrastive learning is inspiring and works impressively well according to the reported experimental results.
- The observation that the fingerprint of a watermarked audio is almost the same as the fingerprint of the original audio is the key to achieve self-containing.
- The experimental results show that the separation performance between benign operation and malicious operation are excellent.

### Weaknesses
It'd be interesting to test the framework's robustness under strong targeted attacks like the ones in [1].

[1] "Can DeepFake Speech be Reliably Detected?." Liu, Hongbin, Youzheng Chen, Arun Narayanan, Athula Balachandran, Pedro J. Moreno, and Lun Wang.

### Questions
The TPR and TNR reported in Table 1 and 2, although super close to 100%, are still not 100%. What should we do in these failure cases?

### Soundness
3

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
4

### Summary
This paper introduces a proactive approach for speech integrity verification, through the combination of multiscale acoustic feature extraction, contrastive fingerprint learning, and segment-wise watermark embedding. It aims to allow for the verification of authenticity directly from a published audio file without needing the original reference. The system’s dual-path design aims to detect tampering while tolerating benign transformations like compression or noise suppression. Extensive experiments demonstrate robust performance in real-world scenarios.

### Strengths
1. The paper proposes a relatively novel approach to address a problem of considerable interest, that of speech watermarking and verification.
2. The formulation of segmentation logic and the learning processes is theoretically sound and verifiable.
3. Well-structured methodology with clear modular design (Speech => feature => fingerprint => watermark).
4. Fingerprinting and watermarking of published content, speech included, is of significant real-life interest and application prospect. The paper’s proposal of a novel approach is therefore of considerable significance.

### Weaknesses
Could be better served to discuss the impact of segmentation lengths to both the effectiveness of watermarking and the robustness to attacks？

### Questions
The watermarking pipeline seems to be reversible and the final watermark reconstructible. While the experiments demonstrated the effectiveness of the proposed approach, a hypothetical attacker aware of the existence of the watermark and the config of the pipeline could potentially replace the original watermark with a newly calculated one. How do the authors envision the mitigation of such attacks?

### Soundness
3

### Presentation
4

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
The paper proposes a speech tampering detection system. Specifically, a self-contained system where the user doesn't have to access the original speech. On a method level, they extract multi-scale features and apply contrastive learning to generate binary fingerprints for the audios. They demonstrate effectiveness on public datasets and a real-world curated dataset.

### Strengths
- The paper is well-motivated, identifying clear limitations in existing passive and proactive speech verification methods and convincingly motivating the need for a self-contained framework.
- The proposed design is conceptually clean and technically solid, integrating multiscale contrastive fingerprint learning with segment-wise watermarking to achieve robustness to benign edits and sensitivity to tampering.
- Experimental results are strong and comprehensive, showing near-perfect detection across multiple datasets and attack types, with clear advantages over established baselines such as RawNet2 and AASIST.
- The system demonstrates practical robustness under realistic distribution settings, including social media compression and re-encoding, indicating readiness for real-world deployment.
- The paper is well-written and thorough, with ablation studies, visualizations, ethical considerations, and reproducibility details that enhance clarity and credibility.

### Weaknesses
Though this paper does seem to be technically solid, I'm a bit concerned about its real novelty and contribution. Specifically, I believe the paper could benefit from a more thorough illustration of previous works on proactive audio watermarking and tampering detection -- when mentioning proactive, the paper is only citing two papers in line, one from 2003 and the other from 2018. This is not new, and the paper is not the first work on so-called "self-contained" detection. The challenges it lists are more or less well-known for the past 5+ years. I'd like to see a better survey of recent relevant works and how this method stands in this stream.

### Questions
- Could you elaborate on how benign vs. malicious pairs are sampled during contrastive training?
- How does this method work with different language/speaker identities? What about different lengths? (like 1 second audio, versus minutes long, is there a way to generalize?)
- The proposed method seems to achieve perfect score on Table 4, compared with poor performance of baselines. I wonder why this happens -- is it just because the dataset setup is easier for the proposed solution? If we further decrease the substitution ratio, how much do we need to get a non-perfect performance?
- Was the multiscale architecture essential—what happens if only one temporal scale is used? I'm not super convinced of the design as it sounds pretty complicated and I didn't see enough experiments highlighting why this exact structure is useful. C.10 provides some numbers, but it might help to explain more about, for example, which types of cases become harder for the model without multi-scale feature, and some intuition about why this is fundamentally useful.

### Soundness
2

### Presentation
3

### Contribution
2

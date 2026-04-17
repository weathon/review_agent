# TVTSyn: Content-Synchronous Time-Varying Timbre for Streaming Voice Conversion and Anonymization

- Decision: Accept (Poster)
- Scores: 2, 6, 8

## Abstract
Real-time voice conversion and speaker anonymization require causal, low-latency synthesis without sacrificing intelligibility or naturalness. Current systems have a core representational mismatch: content is time-varying, while speaker identity is injected as a static global embedding. We introduce a streamable speech synthesizer that aligns the temporal granularity of identity and content via a content-synchronous, time-varying timbre (TVT) representation. A Global Timbre Memory expands a global timbre instance into multiple compact facets; frame-level content attends to this memory, a gate regulates variation, and spherical interpolation preserves identity geometry while enabling smooth local changes. In addition, a factorized vector-quantized bottleneck regularizes content to reduce residual speaker leakage. The resulting system is streamable end-to-end, with <80 ms GPU latency. Experiments show improvements in naturalness, speaker transfer, and anonymization  compared to SOTA streaming baselines, establishing TVT as a scalable approach for privacy-preserving and expressive speech synthesis under strict latency budgets.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper state the static-dynamic mismatch between fixed speaker embeddings and time-varying linguistic content. The proposed TVTSyn model introduces a content-synchronized time-varying timbre representation, complemented by a factorized VQ bottleneck, to balance low latency, naturalness, and privacy.

### Strengths
This paper is well-structured and provide analysis on the effcient of the proposed system.

### Weaknesses
- The paper’s central claim resolving the static-dynamic mismatch via time-varying timbre is not sufficiently novel. Prior work has already explored dynamic speaker conditioning for speech synthesis.
- The dataset employed in experiments is not convince and popular, which make me confuse the correctness of the conclusion.
- Incomplete baseline comparisons with voice privacy challenge baseline systems or other popular speaker anonymization systems.

### Questions
See weaknesses

### Soundness
1

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
4

### Summary
The author proposed a streamable speech synthesizer, TVTSyn, for voice conversion and anonymization. 
The model resolved a mismtach in prior work that linguistic content is time-varying but speaker identity is a static vector. The proposed model aligns the temporal granularity of identity and content relying on TVT, time-varying timbre representation. It also uses a speech content encoder encoder to extract feature that removes residual speaker information and regularize the content space with a factorized vector-quantized bottleneck.
The experiment shows that the model behaves strong compared to multiple baselines and the system is causal end-to-end with <80ms on GPU and <132ms on CPU, which is considered within real-time bounds. A comprehensive study on content and TVT representation and ablation study is also provided. 
Overall, the work has good vision and provide moderate novelty on model architecture and representations to resolve the mismatch between static speaker identity embedding and time-varying timbre representation. The strong empirical results show the effectiveness on model performance including speech naturalness, anonymization and system latency. The limitation of the work is that dataset is english only, evaluation samples size(N=20 for MOS) are limited and the theoretical analysis of the TVT representation could be better discussed.

### Strengths
1. The problem is well framed and it shows good intuition on the mismatch between dynamic input and static speaker embedding. The solution is intuitive by introducing a timbre representation that contains better temporal information.
2. The overall system is well designed and end-to-end streamable. The proposed content encoder introduces a learnable bottleneck with factorized vector-quantization(VQ) that learns discrete, speaker-independent units while preserving linguistic fidelity. Also the time varying timbre representation is consists of a global timbre memory (GTM) that allowing content embedding to attend over the keys to retrieve weighted component using attention, and a combination of gating, interpolation to balance stability and flexibility.
3. The author provided a comprehensive evaluation of the Voice Conversion and Speaker Anonimyzation, and latency analysis.  Also it shows the system is 79 ms latency on GPU and around 132 ms on CPU, achieving a real-time.
4. Good analysis on content representation with tSNE visualization, showing the effectiveness of VQ bottleneck. Good representation on ablation study showing the removal of TVT and VQ causes degradation.

### Weaknesses
1.The discussion for the design of gating/interpolation are mostly based on intuition and empirical results. It would be good to include more theoretical analysis. These innovations on TVT representation and the usage of gating/slerp interpolation are more at a level of improving on top of existing architectures. Yet they are proved effective from experiment results.
2. The MOS tests are based on 20 samples which is limited and may cause bias

### Questions
Could you provide more insights on the design of Factorized VQ and gating/interpolation?

For example, in Section 3.2, you mention that slerp interpolation “respects the hyper-spherical geometry of the embedding space, ensuring smooth trajectories and preserving angular distances” which helps maintain “identity geometry”. Could you clarify or provide theoretical evidence for this claim? 
Also in section 5.2 (c) it shows the difference of before and after applying slerp. How sensitive is the output timbre to the choice of interpolation method (Slerp vs. linear)?

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
The paper introduces TVTSyn, a low-latency voice conversion and anonymization system that replaces the traditional static speaker embedding (x-vectors) with a content-synchronous, time-varying timbre (TVT) representation. The method uses a Global Timbre Memory (GTM), gated interpolation (Slerp), and a factorized VQ bottleneck to align the temporal granularity of speaker identity with linguistic content that are fed into a streamable speech synthesizer for generating anonymised audio. The system achieves <80 ms latency on GPU, ~132ms on CPU, shows improved naturalness, and provides a better privacy-utility trade-off than prior streaming baselines (SLT24, DarkStream, GenVC). Experiments follow the VoicePrivacy Challenge 2024 protocol, reporting reasonable performance in source-target speaker similarity, EER, WER, and perceptual quality metrics. EER in semi-informed case is significantly lower than DarkStream which is not elaborated further. It is potentially due to the content embeddings leaking speaker information. I doubt the claim that content embeddings are speaker-independent and needs to be qualified properly.

### Strengths
- Authors identify a fundamental weakness (static speaker embedding) in the current speaker anonymisation techniques and proposes a well-justified fix via content-synchronous conditioning of the speaker embeddings
- Well-founded experiments on VPC 2024 protocol and ablations confirm benefits across privacy, quality, and latency
- Deployment conditions are kept in mind by demonstrating real-time performance on CPU/GPU under tight latency budgets (<80 ms), relevant for interactive applications
- Clear figures, well-written text, and reproducible evaluation settings
- Concrete future directions are presented that extend the technique significantly

### Weaknesses
- The performance of B1 baseline is mentioned during the analysis but not added to Table 2 for clear comparison
- The gating and Slerp mechanisms are intuitively motivated but not analyzed quantitatively (e.g., contribution to expressivity or privacy).
- Listening tests use a small Mechanical Turk sample (N = 20) without statistical significance analysis or demographic breakdown. A larger cohort of listeners must be recruited (>100) and carefully selected to include demographic variations (age, gender, native/non-native, etc.)
- Authors claim that the content embeddings are speaker-independent and show it through t-SNE plots but do not quantify it through metrics. The claim of speaker-independence needs to be properly verified. One option is to classify speakers directly through content embeddings which might reveal how much speaker information is leaking through them as performed in this paper: https://petsymposium.org/popets/2023/popets-2023-0007.php

### Questions
1. Could the authors provide a quantitative ablation showing how the gating parameter $\alpha_t$ or the number of timbre facets $K$ affects privacy (EER) and quality (NISQA) ?
2. Is the Global Timbre Memory fixed per speaker or updated during fine-tuning ? Would dynamic adaptation hurt anonymity ?
3. How robust is this technique to noisy or reverberant inputs ? does the causal encoder maintain intelligibility under such conditions ?

### Soundness
4

### Presentation
3

### Contribution
3

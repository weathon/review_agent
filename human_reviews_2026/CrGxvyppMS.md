# Data Passports: Confidentially Provable Provenance for Onboarding Verifiable ML

- Decision: Reject
- Scores: 2, 4, 0

## Abstract
Recent advances in ML have leveraged Zero Knowledge Proof protocols to enable institutions to cryptographically commit to a dataset and subsequently prove, to external auditors, the integrity of training and the trustworthiness of the resulting model on the committed data, all while protecting model confidentiality. Such approaches guarantee that the training algorithm which produced a model was computed correctly, but remain vulnerable to pre-commitment data tampering. This is because even if the training algorithm is executed faithfully, an institution can bypass the audit by manipulating the training data. Likewise, data generators may degrade a model’s utility via data poisoning. 

To address this, we introduce tamper-proof Data Passports that bind data to verifiable and confidential proofs of authenticity. We leverage Trusted Execution Environments to issue a certificate of authenticity or ‘passport’ for each data point produced by a generating process. The generating process passes the data and passport to the institution. Then, the institution uses a zero-knowledge proof to verify the validity of the passports to an auditor, as an onboarding step for downstream proofs of training integrity and model trustworthiness. This unlocks cryptographic verification of data provenance throughout the ML pipeline. 

Our experiments demonstrate that we can create tamper-proof passports for images taken by users on their smartphones with a very negligible overhead. Agnostic to data size, a passport can be created at capture time in only 230 ms and consumes just 4.8 KB; thus, it has minimal impact on compute, storage and network usage.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors develop Data Passports - a system that aims to enable verifiable provenance by generating metadata at capture time that attests to the fact that the data was not tampered with. This attestation (denoted the "confidential passport") is then passed downstream to the institution (entity that holds the ML model), who can then employ zero-knowledge proofs to prove to an auditor that each data point is authentic (via hash consistency and signature verification).

### Strengths
The authors identify the issue of self-poisoning - in realistic scenarios, entities may be incentivized to modify their own data in order to circumvent requirements or policy that prevents them from making full use of their data. This is a novel and, to my knowledge, underdressed problem in the literature. To this end, the authors present a system that partially alleviates this concern.

### Weaknesses
System is not end-to-end: Data Passports handles the data generation phase, but training is not handled.  The paper is motivated for this purpose and without it, Data Passports serves no purpose for verifiable training.

It is unclear why TEEs are even necessary for this purpose. The signatures and ZK components of Data Passports rely on computational assumptions. I believe there may be an argument to be made for TEEs as a way to make it harder for the user (the entity collecting data) to circumvent Data Passports entirely, but users might be able to alter their behavior to avoid collecting Data Passports-backed data if they were motivated. Ideally, the paper should make it clear what purpose the TEE serves.

### Questions
1. The security of Data Passports relies heavily on the TEE to ensure that data was not manipulated before the passport is produced, but there is a long line of work demonstrating that TEEs are often breakable (e.g., [1]). How does this impact the security properties of your system?
2. How does the Data Passports system compare to other metadata-based provenance systems like C2PA [2]?
3. I'm not sure how this system resolves the self-poisoning issue. I understand that the system makes it hard for the institution to convince the auditor that manipulated data is from a certified data source. However, once the institution passes verification checks, what's stopping them from training models with slightly (i.e., manipulated) data in step 6? I believe this can be prevented with zero-knowledge proofs of training, but the paper itself mentions this is prohibitively expensive. This should be explained clearly as it's a major motivating point of the paper.

Presentation remarks:
1. add forward references for contributions
2. fix poor quality citations (a, b, etc)
3. Related work on ZKP is missing references
4. Row 38/39: Sentence starting with "Public verification called for" doesn't make sense
5. Row 44/45: form -> forms
6. Row 44 to 49: First half of the second intro paragraph needs to be rewritten - broken grammar, run on sentence, etc.
7. Row 98: consumptions -> consumption
8. Row 104: newborns are issued birth certificates, not necessarily passports
9. Row 122: orx -> or

[1] Van Schaik, Stephan, et al. "Sok: Sgx. fail: How stuff gets exposed." 2024 IEEE symposium on security and privacy (SP). IEEE, 2024.
[2] Rosenthol, Leonard. "C2PA: the world’s first industry standard for content provenance (Conference Presentation)." Applications of Digital Image Processing XLV. Vol. 12226. SPIE, 2022.

### Soundness
1

### Presentation
1

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper tackles a key gap in trustworthy ML pipelines: pre-commitment data tampering, where training data can be manipulated before it’s cryptographically committed for verifiable training. The authors propose Data Passports, a framework combining Trusted Execution Environments (TEEs) and Zero-Knowledge Proofs (ZKPs) to ensure end-to-end verifiable data provenance. In the proposed system, data generated on user devices (e.g., smartphones) is cryptographically signed within a TEE at capture time, producing a "passport" that proves authenticity. Institutions then use ZKPs to prove to auditors that their training datasets consist solely of passported data—preserving confidentiality while ensuring provenance. The authors implement TEE-based passport creation on Android devices and estimate the institution-side ZKP verification costs using parameters from prior work. They report minimal user-side overhead and extrapolate ZKP verification times in the range of tens to hundreds of seconds per image.

### Strengths
+ The paper identifies a critical and timely vulnerability of pre-commitment data tampering in the verifiable ML pipeline
+ The paper effectively integrates TEEs and ZKPs to establish a verifiable, confidential chain of data provenance.
+ The user-side evaluation provides strong evidence on real Android devices that TEE-based data passport generation is efficient and feasible.

Overall, the paper is well written and easy to follow. The idea of combining ZKPs with authenticated data sources is conceptually interesting and, if further developed, could inspire future research in trustworthy and verifiable ML.

### Weaknesses
The paper, while well-intentioned, suffers from several critical weaknesses that undermine its core claims.

- My major concern arises from the evaluation of the core ZKP component. It appears the authors did not implement their ZKP circuit but instead chose to "extrapolate costs" from a prior paper [Frigo & abhi shelat, 2024]. Since ZKP costs depend heavily on circuit structure, memory layout, and batching strategy, this extrapolation is not a reliable proxy for end-to-end feasibility. The reported prover times (40–290s per image) imply tens of days of computation for even modest datasets (10^5 images). This may render the system infeasible for its intended ML onboarding use case in practice. A minimal working prototype, even for small circuits, would have strengthened the paper's empirical foundation.

- A second major concern lies with the security, which relies on overly optimistic trust assumptions and omits key threats related to the TEE use. The paper emphasizes TEE efficiency and isolation benefits but overlooks their accompanying risks.
  1. The framework does not consider malicious code inside the TEE. However, an institution-controlled application could (and it is possible) inject bias or alter data within the TEE before signing, yielding a valid passport for tampered input. This could directly undermine the claimed protection against "self-poisoning".
  2. The passport only attests to where and when a capture occurred, not what was captured. A user could take a TEE-certified photo of a poisoned or synthetic image displayed on a monitor (aka analog hole attack); the system would treat it as authentic and this easily bypasses the protection against "user data manipulation".
  3. The threat model implicitly assumes a "perfect" TEE. It omits adequate discussion of real-world hardware attacks like side-channels, fault-injection, or rollback attacks, which is a significant oversight for a system whose entire trust is anchored in this hardware.

- Third, it reads as though the paper’s main technical contribution lies in this TEE–ZKP “co-design,” but it is presented more like an intuitive A+B combination than a genuinely integrated system. Prior work has already explored TEE-assisted provenance (e.g., Truepic, ProvCam, Vronicle) and ZKML frameworks in depth. What's missing here is a discussion of the practical difficulties created by this very integration itself—issues that are crucial to assessing whether the proposed approach can actually work in practice.
  1. The paper frames the ZKP cost as a simple "extrapolation", but it's a direct and fatal consequence of this design. Forcing the ZKP to prove a SHA-256 hash over the entire image file (as TEE-signing requires) is exactly why the cost is tens of seconds per image. The "co-design" itself is what makes the system a bit unscalable.
  2. The current passport only binds a raw data file $Sign_{sk}(SHA256(data))$. This is fundamentally incompatible with any real-world ML pipeline. How are data labels authenticated? What about the necessary pre-processing, data cleaning, or feature extraction steps that all ML requires? Any of these steps would invalidate the passport.

- Minor:
  1. Line 045, "various form of" -> "various forms of".
  2. Line 266, "proceeds capturing" -> "proceeds to capture".
  3. Line 352, "an passport" -> "a passport".
  
Overall, because it avoids these hard questions, the "ZKP-CI" primitive remains largely a conceptual sketch. It lacks formal security definitions, proofs, and practical validation of scalability, which limits its value as a concrete technical contribution.

### Questions
1. Could the authors elaborate on the decision to extrapolate ZKP costs rather than implementing even a simplified prototype? This makes the core feasibility claim difficult to verify.
2. Relatedly, how would the performance or complexity change if the ZKP were redesigned to avoid hashing the entire image (e.g., through commitments on feature-level data or chunked proofs)? Exploring such variants could clarify whether the scalability bottleneck is fundamental or merely an artifact of the current design.
3. How does the framework defend against a certified-but-malicious app that tampers with data inside the TEE before signing? If this threat is considered out of scope, could the authors clarify why it is not a concern for the intended use cases?
4. How does the system address the "analog hole" attack for user data poisoning? If this vector is out of scope, a discussion of its implications would help clarify the system’s real-world robustness.
5. Could the authors comment on why real-world TEE vulnerabilities (e.g., side-channels, fault-injection) were omitted from the threat model, given they directly impact the system's root of trust?
6. Since the passport currently signs only raw data, how does the system authenticate downstream processing steps—such as labeling, normalization, or feature extraction—that are essential for ML training? Without this, how can the provenance guarantee extend to the actual training dataset?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
0

### Rating Number
0

### Confidence
4

### Summary
The paper proposes “Data Passports” as a mechanism to certify training-data provenance for machine‐learning pipelines. Each user‐generated data item is issued a cryptographic “passport” at creation time (within a Trusted Execution Environment, TEE) encoding provenance metadata. The institution then uses a zero-knowledge proof (ZKP) to show an auditor that it only used data items bearing valid passports — thus preventing both institutional data manipulation (post-commit tampering) and user-side poisoning of datasets. The authors claim this extends verifiable ML beyond just model training to *data authenticity*, and provide a prototype implementation (on Android phones + ZKP verification) to show low overhead.

### Strengths
Tackles a timely and relevant problem: data provenance is increasingly recognised as a weak link in trustworthy ML. 

The authors include a prototype and some empirical overhead measurements (on Android devices, CPU/memory/battery of passport creation) which strengthens the work’s engineering credibility.

The intuitive analogy of a “passport for data” helps make the idea accessible, which is helpful given the intersection of ML, security, and hardware trust.

### Weaknesses
Justification of ZKP & TEE is weak. The paper does not convincingly explain *why* a full zero‐knowledge proof is required, or *why* a TEE is needed rather than a simpler approach (e.g., device signs the hash of the sample, institution publishes signature, verifier checks signature). If the passport metadata are non‐sensitive, then a simpler digital signature scheme might suffice. The use of ZKP and TEE therefore feels like over‐engineering or lacking sufficient motivation/discussion.

Bias and provenance vs. authenticity trade‐off omitted. While the mechanism aims at authenticity, it doesn’t address *bias*, representativeness or *quality* of the data. The system might certify “this sample was captured by a trusted device at time T” but that says nothing about whether the sample is training‐appropriate, non-biased, or non‐poisonous (in a semantic sense). The authors should more clearly delineate what the passport does *and does not* guarantee.

Security model is insufficiently detailed. The paper lacks a formal adversary model, threat‐analysis table, or proof sketch of soundness. It assumes the TEE, certificate authority (CA), and device registration are trusted, but does not articulate what happens if keys are compromised, certificates forged, TEE is subverted, or side‐channels exploited. Without this, the “tamper‐proof” claim is overstated.

Clarity and writing issues. Some language is informal to me (e.g., on p.4: “I uses a zero‐knowledge proof to verify to V that …”), which detracts from readability and professionalism. Some design assumptions and implementation details are glossed too lightly. 

Scalability and deployment issues under-explored. While user‐device overhead is measured, the institution-side proof generation and verification overhead for large datasets appears to grow linearly (each sample’s metadata must be proven/verified). The paper acknowledges this but does not give quantitative limits (e.g., how many millions of samples is feasible). Also, the ecosystem dependence (every device must support TEE signing; every source must be registered; legacy data without passports must be handled) is only briefly discussed.

### Questions
My questions are mainly listed in the part of weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

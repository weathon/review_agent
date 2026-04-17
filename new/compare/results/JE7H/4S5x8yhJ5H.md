---
job_id: 52472c82-302a-4855-9989-3dfbb4f4ba3f
agent_id: ICLR_main_2026_1
status: COMPLETED
filename: 4S5x8yhJ5H.pdf
paper: Vibeface - Video and Image Biometric Dataset for Evaluation of Faces
main_score_norm: 0.4
desk_reject: false
---
# Desk Rejection Assessment:
## Paper Length  
Pass ✅.

## Topic Compatibility  
Pass ✅. The paper presents a new facial biometric dataset plus benchmark tasks for face detection and verification, which clearly falls under “datasets and benchmarks” and “representation learning for computer vision” within ICLR’s scope.

## Minimum Quality  
Pass ✅. The paper is in English and has all essential components for a dataset/benchmark paper: Abstract, Introduction, Related Work, Dataset/Methodology (Section 3), Experiments & Results (Section 4), and Conclusions (Section 5). The experimental methodology is basic but not fatally flawed; there is sufficient detail to understand what was done.

## Prompt Injection and Hidden Manipulation Detection  
Pass ✅. I did not find any hidden instructions or manipulation attempts targeting automated reviewing systems in the provided content.

---

# Expected Review Outcome:

## Summary

The paper introduces VIBEFACE, a facial biometric dataset aimed at evaluating face verification in eKYC-like scenarios. It includes 2,250 high-resolution still images and 1,550 short videos from 50 participants, collected over five sessions that vary lighting, eyeglasses, and capture scenarios, with detailed demographic balance across gender, race, and age. The authors additionally provide baseline evaluations for face detection (MTCNN, RetinaFace, MediaPipe) and face verification (ArcFace, MagFace) under different scenarios and demographic slices.

## Strengths

1. **Clear and well-structured dataset design with realistic eKYC scenarios.**  
   The paper systematically defines 18 scenarios (Section 3.2) covering standardized poses, selfie photos, selfie videos, and eKYC-style verification videos with specific actions (head rotations, blinking, expression change, partial occlusion, etc.). **Figure 2** and **Figure 3** are effective in visually illustrating the different photo and video scenarios (e.g., left/right profiles, head rotations, blinking, occlusions), which helps readers understand the practical relevance of the acquisition protocol for eKYC workflows.

2. **Demographic balancing is unusually explicit and transparent.**  
   The dataset is carefully balanced across gender (25 female, 25 male) and four self-identified racial categories (13 African, 13 Caucasian, 12 East Asian, 12 South Asian), and spans ages 18–69. **Figure 1A** (demographic pyramid) and **Figure 1B** (ethnic distribution) concretely demonstrate that the distribution is not skewed toward a single group, which is a real problem in many existing face datasets. The paper is explicit about following ISO standards for age structure (Section 3.1), which is a plus for downstream fairness studies.

3. **Ethical and legal compliance is front and center.**  
   Section 3.4 carefully details GDPR- and AI Act-compliant collection, informed consent, anonymization, and controlled-access release for non-commercial research. This is important given the recent withdrawal of major face datasets due to consent and privacy issues, and it is good to see this addressed explicitly rather than as an afterthought.

4. **Useful baseline analyses across conditions and demographics.**  
   Section 4 provides baseline results for face detection and verification broken down by scenario, session, gender, age group, and racial category. **Table 3** systematically summarizes face detection success rates for MTCNN, RetinaFace, and MediaPipe across off-angle views (OAV), frontal views (FV), dynamic scenarios (12–16), and demographic slices. **Table 4** does the same for ArcFace and MagFace verification performance, including the effect of eyeglasses (Session C) and weak natural light (Session E). This level of breakdown is valuable for researchers interested in bias and robustness analysis.

5. **Realistic capture hardware and conditions.**  
   Data are captured on common consumer smartphones (Xiaomi Redmi Note 13, iPhone 13, Samsung Galaxy A35 5G) under multiple lighting setups (artificial, flash, natural, weak natural) and with/without eyeglasses, as detailed in Section 3.3. **Table 2** concisely shows which scenarios occur in which sessions and which camera (front/back) is used, making it relatively easy to reason about what data exists for which condition.

6. **Responsible positioning in relation to problematic prior datasets.**  
   The Related Work section correctly notes that important large Internet-scraped datasets (MS-Celeb-1M, VGGFace2, MegaFace) have been withdrawn due to consent/privacy issues and attempts instead to position VIBEFACE among controlled, ethically collected datasets like MOBIO, MobiBits, WMCA/HQ-WMCA, and SOTERIA. **Table 1** is a helpful high-level comparison of these datasets in terms of IDs, presence of photos/videos, eKYC-style content, demography information, and balance dimensions.

## Weaknesses

1. **Scale and coverage are modest for modern face/representation learning needs.**  
   The dataset has only 50 identities (Section 3) and 1,550 videos total, which is tiny relative to current practice in face recognition and fairness evaluation. This is acknowledged implicitly by comparing with datasets like MOBIO and SOTERIA in **Table 1**, but the paper does not seriously confront how this small scale limits the sort of representation learning or bias analyses that can be meaningfully conducted. For example, any demographic disparity estimates will have high variance and limited generalizability. The paper should be much clearer and more honest about what statistical analyses are and are not realistic with 13 African and 12 East Asian participants, etc.

2. **Very shallow evaluation protocol for face verification, with questionable thresholding.**  
   In Section 4.2, verification success is defined as a frame-level similarity score > 0.5 for ArcFace and MagFace, with no justification of this threshold, no ROC/EER curves, and no calibration or cross-validation. Using a fixed arbitrary threshold across models, sessions, and scenarios is not standard practice in biometrics research and is particularly problematic for MagFace, whose score distributions differ from ArcFace by design. **Table 4** then reports only “percentage of frames correctly authenticated”, but we have no False Accept Rate / False Reject Rate trade-off, no per-identity analysis, and no notion of impostor trials at all. Essentially, the experiment measures how often a genuine user is above an arbitrary threshold, but not actual verification performance in a biometric sense. This substantially limits the value of the presented results and may mislead readers about model behavior in realistic verification settings.

3. **Benchmark tasks are limited and not tightly connected to representation learning questions.**  
   Both benchmark tasks (Sections 4.1 and 4.2) use off-the-shelf pre-trained models with no training on VIBEFACE and no comparison of different training strategies or representation learning approaches. There are no experiments that show how training or fine-tuning on VIBEFACE affects generalization, bias, or robustness, which would make the dataset much more relevant to ICLR’s core audience. As it stands, the dataset is primarily demonstrated as a testbed for pre-existing detectors/recognizers, not as a driver for new representation learning research.

4. **Detection evaluation is arguably too easy and not clearly defined.**  
   Section 4.1 states that detection success is “percentage of frames in which a face was successfully detected” but never defines a bounding box quality criterion (IoU threshold, minimum size, etc.). This is critical for interpreting **Table 3**, where RetinaFace and MediaPipe reach 100% or near-100% detection in almost every setting, including off-angle views and dynamic scenarios, while MTCNN lags substantially. The fact that RetinaFace has 1.000 detection rate for all OAV and FV scenarios across sessions suggests that the metric is extremely lenient or that the dataset is not challenging for modern detectors. Without a clear IoU-based criterion or some annotation reference, the reported detection numbers are difficult to trust or compare with other work.

5. **Limited and somewhat inconsistent fairness analysis given the claims.**  
   A major motivation of the paper is fairness and demographic robustness, yet the analysis is very cursory. For detection (Table 3), the demographic section collapses the four races into three columns (Afr., Cauc., EA) and inexplicably omits the SA (South Asian) group from the table, despite the dataset explicitly including South Asian participants (Section 3.1 and Figure 1b). For verification (Table 4), there is a racial column for SA, but the conclusions in Section 4.2 are extremely brief (“Both models performed slightly worse on the Caucasian subgroup”) and do not quantify effect sizes or statistical significance. There is no attempt to compute group-wise error rates at matched operating points (e.g., equal FAR), which is standard in fairness-oriented biometric work. This mismatch between strong fairness motivation in the Introduction and very shallow fairness analysis undermines the claimed significance.

6. **Positioning against directly related eKYC datasets is incomplete.**  
   While **Table 1** compares VIBEFACE to several biometric datasets, the paper omits recent work specifically focused on eKYC and related video-based verification for fraud or deepfake detection. For example, an eKYC-focused dataset that includes deepfake/injection-style attacks is directly relevant and should be contrasted with VIBEFACE in terms of scope (bona fide vs. attack), capture protocols, and intended tasks. Without these comparisons, the “to the best of our knowledge” claim that no publicly available dataset includes authentic eKYC-style videos looks shaky and at least under-argued.

7. **No explicit train/validation/test partitioning or recommended protocol.**  
   For a benchmark dataset, the paper should provide recommended splits to avoid inadvertent overfitting or identity leakage across train/test. Section 4 simply uses a single frontal flash image from Session B as a reference and uses all other data as queries, but there is no suggestion of how to create a standardized evaluation protocol for future methods (e.g., specific identities for training vs. testing, or specific sessions for enrollment vs. verification). This makes reproducible comparison across papers harder, and reduces the immediate usability of the dataset as a benchmark.

8. **Methodological clarity issues in the evaluation setup.**  
   Several aspects of the experimental methodology remain under-specified:
   - For videos, Section 4.1 states that frames are sampled at 6 fps, but Section 4.2 does not clarify whether verification is done independently on each frame or aggregated over a video (it appears to be frame-level). This is particularly important when reporting percentages in **Table 4**, since longer videos will contribute more frames.  
   - There is no description of pre-processing (face alignment, cropping) for the verification experiment, yet this can strongly influence ArcFace/MagFace scores.  
   - The paper does not clarify whether the detectors used in Section 4.1 are also used to crop faces for verification in Section 4.2, or whether another detector/pre-processing is applied.  
   These gaps reduce the reproducibility and interpretability of the reported numbers.

9. **Mathematical and metric specification is very light and somewhat sloppy.**  
   Although this is a dataset paper, the small amount of math that is present is under-specified. For face verification, the “similarity score” in Section 4.2 is not defined: is it cosine similarity of normalized embeddings, Euclidean distance converted somehow, or MagFace’s quality-aware score? The exact function \( s(x_{\text{ref}}, x_{\text{query}}) \) and its transformation into a decision via threshold \( \tau = 0.5 \) should be explicitly written, for example:
   \[
   s(x_{\text{ref}}, x_{\text{query}}) = \cos(f(x_{\text{ref}}), f(x_{\text{query}})),
   \quad \hat{y} = \mathbb{1}\{ s(x_{\text{ref}}, x_{\text{query}}) \ge \tau \}.
   \]
   Without this, it is difficult to reason about how scores behave across models or to reproduce the performance in **Table 4**. Similarly, for detection, a formal definition like:
   \[
   \text{Detected} = \mathbb{1}\{ \max_{b \in \mathcal{B}} \text{IoU}(b, b^*) \ge \alpha \}
   \]
   (with specified \(\alpha\) and annotation \(b^*\)) should be provided. At present there are no annotations or IoU thresholds discussed at all.

10. **Some inconsistencies and missing details in tables and figures.**  
    - In **Table 3**, SA (South Asian) is mentioned in the caption but absent from the race columns, which list only Afr., Cauc., EA. This is confusing and contradicts the claim of racial balance.  
    - **Table 1** is spread awkwardly across rows (dataset name and citation separated), and many table cells are empty, which makes it harder than necessary to parse.  
    - A minor but telling issue is that in Table 2 the header for scenarios “1–5, 6–10, 11, 12–18” is split across two rows in a confusing way; while not a scientific flaw, it adds friction given the otherwise clear exposition.

Overall, while the dataset itself is carefully designed and ethically collected, the experimental section is relatively weak and underspecified for a main-track ICLR submission, and the small scale significantly limits what representation learning or fairness conclusions can be drawn.

## Potentially Missing Related Work

1. **Felouat, H., Nguyen, H., Le, T. (2024). “eKYC-DF: A Large-Scale Deepfake Dataset for Developing and Evaluating eKYC Systems.”**  
   - Relevance: This work targets eKYC specifically, providing a dataset for evaluating eKYC systems, including attack scenarios (deepfakes). It is directly related to the stated goal of VIBEFACE as a dataset for eKYC-like workflows.  
   - How to integrate: It should be discussed in Section 2 (Related work) in the context of datasets that support eKYC evaluation, clarifying that VIBEFACE focuses on bona fide samples and realistic capture, while eKYC-DF focuses on attack content. It would also be appropriate to mention it around the claim in the Introduction and Conclusion that “to the best of our knowledge, there are no publicly available datasets that include authentic eKYC-style facial videos”, since eKYC-DF is at least tangentially overlapping in scope.

2. **Hanawa, G., Ito, K., Aoki, T. (2024). “Face image de-identification based on feature embedding.”**  
   - Relevance: This paper proposes methods to de-identify face images while maintaining utility, which is directly relevant to privacy-preserving handling of biometric datasets like VIBEFACE.  
   - How to integrate: It would fit naturally into Section 3.4 (Ethical considerations and privacy) or Section 2, to contextualize possible future extensions where VIBEFACE images could be released in de-identified form, or where such techniques could be evaluated using this dataset.

3. **Zeinstra, C. G., Veldhuis, R. N. J., Spreeuwers, L. J. (2017). “ForenFace: a unique annotated forensic facial image dataset and toolset.”**  
   - Relevance: ForenFace is another carefully curated, controlled facial image dataset with a strong focus on annotation and real-world operational conditions (in that case, forensic), similar in spirit to VIBEFACE’s focus on eKYC.  
   - How to integrate: It should be briefly compared in Section 2 as another example of a specialized, ethically collected facial dataset, helping to more precisely position VIBEFACE among existing “operation-specific” datasets.

## Questions

1. **Verification protocol and thresholding.**  
   - How exactly are similarity scores computed for ArcFace and MagFace? Are embeddings L2-normalized and cosine similarity used?  
   - How was the global threshold of 0.5 chosen, and how sensitive are the reported percentages in **Table 4** to this threshold? Could you provide at least one ROC or DET curve and report EER or a standard operating point (e.g., FAR = 1%)?

2. **Presence or absence of impostor trials.**  
   Section 4.2 appears to consider only genuine trials (participant vs. their own reference image). Did you consider any impostor trials (participant vs. other participants’ reference images)? If yes, how many and how were they sampled? If not, what is your view on how practitioners should use VIBEFACE to evaluate full verification systems, including impostor error rates?

3. **Detection evaluation details.**  
   - Are there ground-truth bounding boxes or landmarks for faces in VIBEFACE? If so, what IoU threshold or criterion is used to judge whether a detection is “successful” in **Table 3**?  
   - If there are no annotations, what heuristic determines that a detector “found a face” (e.g., at least one bounding box in the frame)? This is important to interpret the near-perfect RetinaFace/MediaPipe results.

4. **Recommended train/validation/test splits and protocols.**  
   Do you intend to release recommended splits (e.g., per-identity or per-session) and standardized evaluation scripts? If so, could you outline them concretely, including how to partition sessions, scenarios, and demographics? This would significantly raise the dataset’s benchmarking value.

5. **Fairness analysis depth.**  
   - For the demographic breakdowns in **Table 3** and **Table 4**, are the differences across groups statistically significant? Could you provide confidence intervals or at least rough variance estimates over identities?  
   - Why is the South Asian group not shown in **Table 3**, even though it appears in the dataset and in **Figure 1B**?

6. **Potential for PAD and deepfake evaluation.**  
   You mention in the Conclusion that VIBEFACE could be useful for presentation attack detection and injection/deepfake detection. Can you elaborate more concretely how you envision constructing PAD or deepfake benchmarks using this data, considering that the current dataset only contains bona fide samples?

Clarifying these points and tightening the evaluation methodology could significantly improve my confidence in the scientific value of the benchmark section.

## Flag For Ethics Review

- Yes, Privacy, security and safety  
- Yes, Potentially harmful insights, methodologies and applications  
- Yes, Responsible research practice (e.g., human subjects, data release)  

## Details Of Ethics Concerns

This is a biometric dataset with identifiable facial images and videos, which raises inherent privacy and misuse risks (e.g., unauthorized re-identification, surveillance, or deployment in non-consensual contexts). Section 3.4 and 3.5 indicate GDPR- and AI Act-compliant procedures, anonymization, informed consent, and a non-commercial controlled-access license. Nonetheless, given the sensitivity of face biometrics and the public release of data, an ethics review seems appropriate to double-check:  
- Whether the consent forms explicitly cover global research use and potential long-term storage;  
- Whether the access control and license enforcement mechanisms are sufficiently robust;  
- Whether there are mitigation strategies for potential misuse (e.g., scraping of released images, model re-identification attacks).  

These are not indications of misconduct, but reflect the inherently sensitive nature of the dataset.

## Soundness Rating

2: fair.  
The dataset itself is collected in a sound and well-documented way, but the evaluation methodology for detection and especially verification is under-specified (no clear metric definitions, arbitrary thresholds, no impostor trials), which weakens the technical soundness of the benchmark results.

## Presentation Rating

3: good.  
The paper is generally well written and organized; figures (especially Figures 1–3) are helpful, and the dataset description is clear. However, some tables are confusing (e.g., Table 1 formatting, missing SA column in Table 3), and key experimental details are missing from the text.

## Contribution Rating

2: fair.  
The ethical, demographically balanced eKYC-style dataset is useful, but the scale (50 identities) is limited and the demonstrated benchmarks are relatively shallow. The work is more at the level of a solid workshop dataset paper than a strong ICLR main-track contribution.

## Overall Rating

4: Marginally below the acceptance threshold. But would not mind if paper is accepted.  
The dataset is thoughtfully designed, ethically collected, and clearly described, with realistic eKYC scenarios and useful demographic balancing that will be of interest to parts of the biometrics community. However, the small scale and the weak, under-specified experimental protocol limit its impact for representation learning research at ICLR. Strengths in ethical design and clarity are offset by methodological gaps in the evaluation, leading me to lean slightly negative for the main track while recognizing that the dataset itself is valuable.

## Reviewer Confidence

4: confident.  
I am familiar with facial biometric datasets, fairness issues, and verification/detection evaluation, and I carefully examined the methodology and tables. Some details are missing (by the authors) rather than unclear to me, so my confidence in the assessment is high.
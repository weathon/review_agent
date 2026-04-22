# PHUMA: Physically-Grounded Humanoid Locomotion Dataset

- Avg Score: 5.00
- Decision: Reject
- Scores: 6, 4, 2, 8

## Abstract
Motion imitation is a promising approach for humanoid locomotion, enabling agents to acquire humanlike behaviors. Existing methods typically rely on high-quality motion capture datasets such as AMASS, but these are scarce and expensive, limiting scalability and diversity. Recent studies attempt to scale data collection by converting large-scale internet videos, exemplified by Humanoid-X. However, they often introduce physical artifacts such as floating, penetration, and foot skating, which hinder stable imitation.
In response, we introduce \textbf{PHUMA}, a \textbf{P}hysically-grounded \textbf{HUMA}noid locomotion dataset that leverages human video at scale, while addressing physical artifacts through careful data curation and physics-constrained retargeting.
PHUMA enforces joint limits, ensures ground contact, and eliminates foot skating, producing motions that are both large-scale and physically reliable.
We evaluated PHUMA in two sets of conditions: (i) imitation of unseen motion from self-recorded test videos and (ii) path following with pelvis-only guidance. In both cases, PHUMA-trained policies outperform Humanoid-X and AMASS, achieving significant gains in imitating diverse motions. PHUMA will be publicly released to support future research.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This submission introduces PHUMA, a physically-grounded humanoid locomotion dataset created from human videos with a physics-constrained retargeting pipeline that enforces joint limits, reliable ground contact, and removes foot-skate. Trained policies on PHUMA outperform counterparts trained on AMASS and Humanoid-X in motion imitation and path-following tests. The authors plan to release the dataset to the community.

### Strengths
- A large-scale (73h), curated, physics-aware dataset that  addresses well-known artifacts in video-derived motion (e.g., joint limits, penetration, floating, foot-skate). This provides practical value to the community.
- The method is sound. The proposed PhySINK extends SINK to deal with joint limits, ground contact, and foot skating.
- The evaluation is principled and thorough. Evaluations show consistent gains over AMASS and Humanoid-X across two use cases (unseen-video imitation, pelvis-only path following).

### Weaknesses
- Related works: Please compare/relate to recent efforts that also scale human to humanoid data, e.g., H2O/Human2Humanoid, ASAP, Humanoid Policy ~ Human Policy.
- How does the proposed method handle non-planar ground, such as ground with height changes or stairs?
- Physically invalid motions are removed but not refined. This could reduce the diversity of the data.

### Questions
- what is the data distribution in terms of scene geometry, material, and semantics diversity (e.g., indoor vs outdoor, height variation, ground material)
- To what extent is retargeting tied to a specific humanoid morphology? What is the distribution over body shape and heights.
- How is foot contact detected? How reliable is that?
- It would strengthen the paper to quantify how each physics constraint (contact, joint limits, anti-skate) contributes to downstream policy quality.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
To address the data scalability issue of humanoid locomotion, the authors proposed PHUMA. With carefully curated physics-aware motion filtering, retargetting, and learning, PHUMA managed to effectively enlarge the available data scale for humanoid. The effectiveness of PHUMA is validated on unseen motion imitation and path following. Impressive performance is achieved.

### Strengths
- The curated PHUMA is superior to existing efforts in scale and diversity.

- The proposed pipeline seems reasonable and practical.

- The proposed pipeline and dataset could serve as valuable resources for humanoid learning.

### Weaknesses
- A noticeable characteristic of PHUMA is its heterogeneous data sources. Extending the existing data quality analysis and performance analysis to data from different sources would be preferred. A separate evaluation of mocap-sourced and video-sourced PHUMA would be helpful.

- Comparisons between the proposed physink and related works, like GMR, are missing. 

- Though the comparison of heuristic metrics in Table 2 is straightforward, a missed chance would be investigating the relationship between these physical artifacts and their influence on the humanoid learning procedure.

- The physics-based filtering hasn't been well analyzed with ablation studies.

- Many details are missing, some of which are closely related to the data quality and algorithm performance.

- No video demonstration is attached, which is important for identifying the data quality. 

- No hardware deployment results are provided.

### Questions
- In Table 1, most of the video datasets only have a frame rate of 30FPS. How would the low-pass filter with a cutoff frequency of 30Hz, as mentioned in A.1.1, work for these data? 

- How were the limits and thresholds in Table 7 decided? How many motion sequences were filtered out for each term? Also, how were the corresponding influences on the performance?

- If the curation pipeline is applied to LAFAN1 and AMASS, would the corresponding performance be improved?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper presents PHUMA, a large-scale physically grounded humanoid motion dataset designed for stable imitation learning in simulation. It addresses the problem that motion data extracted from Internet videos, such as Humanoid-X, often contain artifacts like floating, penetration, and joint-limit violations that degrade physics-based policy training. PHUMA introduces a physics-aware curation process that filters motions for contact and balance consistency, and a physics-constrained retargeting algorithm, PhySINK, that enforces non-floating, non-penetration, and non-skating constraints. The resulting dataset contains about 76k motion clips covering 73 hours of locomotion data. Experiments on the Unitree G1 and H1-2 humanoids in Isaac Gym show that policies trained on PHUMA achieve substantially higher imitation success and physical stability than those trained on LaFAN1, AMASS, or Humanoid-X.

### Strengths
1. The paper focuses on physically grounded humanoid motion, addressing a gap in large-scale imitation datasets where stability and contact consistency are often ignored.

2. The proposed PhySINK pipeline is technically solid and produces cleaner motion data with fewer artifacts than Humanoid-X.

3. The experiments are comprehensive within simulation, demonstrating clear quantitative improvements on multiple humanoid platforms and providing a useful dataset that can benefit future physics-based imitation learning research.

### Weaknesses
1. While the paper presents a clean and useful dataset, its novelty appears somewhat limited relative to recent works that also improve over Humanoid-X through physics-aware retargeting. Methods such as ASAP (RSS 2025), KunfuBot (NeurIPS 2025), and GMR have already introduced substantial innovations in motion retargeting and data cleaning—ASAP integrates RL-based physical simulation during retargeting, KunfuBot applies extensive filtering for realistic contacts, and GMR focuses specifically on retargeting fidelity—whereas PHUMA mainly improves data quality through curation and constraints. Compared with those works, the contribution here lies primarily in dataset refinement rather than methodological advance. 

2. In addition, the dataset only covers locomotion, leaving out other interaction behaviors, and all results are limited to simulation without any hardware validation. There is no qualitative simulation videos for reference, which makes it difficult to assess the realism of the resulting motions.

### Questions
The imitation results are reported on both Unitree G1 and H1-2. Are the same PHUMA motions directly retargeted to each robot, or are robot-specific shape/scale adjustments applied?

Are there qualitative examples where PhySINK still fails?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
PHUMA introduces a new dataset of physically-grounded humanoid motions reconstructed then processed from videos of humans. The pipeline extracts human motion from RGB video then retargets it using a proposed Physically-grounded Shape-adaptive Inverse Kinematics (PhySINK) procedure. For policies trained on the PHUMA dataset, the authors observe higher success rates than corresponding policies trained on AMASS and Humanoid-X.

### Strengths
- Data curation and processing pipeline is well-justified and thoroughly explained.
- PhySINK has good quantitative comparison to SINK and Mink's IK as appropriate baselines.
- Method identifies real problems with existing video-to-humanoid motion pipelines that lead to poor downstream performance during RL training and deployment.
- The dataset is relatively large, comparing favorably against popular datasets like AMASS
- Data pipeline validated on multiple humanoid robot form factors.

### Weaknesses
- 1.2x success rate improvement over AMASS isn't very dramatic.
- "Physically grounded" in this paper means the motions have kinematic plausibility as defined by heuristics and loosely verified by downstream sim RL training. I think this is somewhat misleading wording, since I would interpret "physically grounded" to mean that it is simulated dynamically in the data processing loop.
- A sim-to-real deployment would be a stronger validation of the data pipeline.

### Questions
- Why is VIBE used as opposed to a newer model like VIMO?

### Soundness
3

### Presentation
3

### Contribution
3

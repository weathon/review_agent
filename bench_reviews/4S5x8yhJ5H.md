## Summary
This paper introduces VIBEFACE, a novel multimodal dataset for evaluating face verification systems, specifically targeting electronic Know Your Client (eKYC) scenarios. The dataset comprises 2,250 images and 1,550 videos from 50 demographically balanced subjects, captured under varied lighting conditions and with specific eKYC action sequences. The authors demonstrate its utility through preliminary benchmarks on face detection and verification tasks.

## Strengths
- **Ethically Sourced and Demographically Balanced Dataset:** The data collection adheres to stringent ethical standards (GDPR, AI Act compliance, informed consent) and achieves commendable balance across gender (25:25), four racial categories, and a wide age range (18-69). This addresses a critical gap in responsibly sourced biometric data.
- **Novel eKYC-Specific Video Scenarios:** VIBEFACE is the first publicly available dataset to include video sequences explicitly designed to mimic real eKYC verification workflows (e.g., head rotation, blinking, expression changes), filling an identified application gap.
- **Structured and Well-Documented Design:** The dataset is methodically constructed with clear scenarios (standardized photos, selfies, action videos) and sessions (varying lighting, presence of eyeglasses), providing rich, annotated data for controlled analysis of robustness factors.

## Weaknesses
### Major:
- **Limited Dataset Scale Undermines Benchmark Claims:** The dataset contains only 50 unique identities. While sufficient for initial analysis, this scale is orders of magnitude smaller than modern face recognition benchmarks (e.g., WebFace260M) and limits the statistical power for robust fairness analysis and generalizability claims. The paper's assertion that VIBEFACE provides a "comprehensive" resource and a "new benchmark" is overstated given this constraint.
- **Overly Simplistic and Non-Standard Verification Evaluation:** The face verification benchmark (Sec. 4.2) uses a fixed similarity threshold (0.5) for both models and reports only frame-level verification rates. This is not standard practice; the field relies on metrics like TAR@FAR, EER, or ROC curves to evaluate the trade-off between false acceptance and rejection. The chosen protocol obscures the true difficulty of the task and prevents meaningful model comparison. Furthermore, the evaluation treats videos as bags of independent frames, failing to leverage the temporal dimension or simulate a realistic eKYC matching pipeline.
- **Insufficient Analysis to Demonstrate Unique Value:** The paper lacks experiments that concretely show VIBEFACE introduces challenges not captured by existing datasets. There is no comparative benchmarking (e.g., evaluating the same model on VIBEFACE vs. SOTERIA) to quantify the added difficulty of its eKYC scenarios or varied conditions. The claimed suitability for research in presentation attack detection (PAD) or liveness detection is also not validated with any experiments.

### Minor:
- **Under-explored Impact of Acquisition Variables:** The dataset was collected using three different smartphone models, but the analysis does not isolate the impact of sensor variability on performance. A breakdown of results by capture device would strengthen the claim of evaluating "cross-device variability."
- **Lack of Formal Fairness Metrics:** While performance is broken down by demographic groups, the analysis does not compute established fairness metrics (e.g., equalized odds difference, demographic parity). This limits the paper's contribution to "advancing fair... benchmarking."

### Trivial:
- The temporary data access link and password provided for review are functional, so access for evaluation is not an issue.

## Nice-to-Haves
- A power analysis or discussion of the statistical confidence limits for subgroup comparisons given the sample size of 50 subjects.
- Visualization of failure cases (e.g., false non-matches) across different demographics and challenging scenarios to provide intuitive insight into remaining problems.

## Removed Points
*These points are flagged to be removed or were not included as weaknesses, treat them with caution.*
- **Strength - "Clear Motivation":** While valid, this is a generic strength applicable to many papers that identify a research gap.
- **Weakness - "Ambiguous Data Access and Licensing":** The paper specifies a controlled-access license and a process via a Research Data License Agreement. This is a concrete plan, not vagueness. The provided temporary link works for review.
- **Weakness - "Incomplete Exploration of Downstream Applications":** The paper's scope is to introduce and benchmark the dataset for core verification tasks. Demanding evaluation of PAD, age estimation, or emotion recognition is scope creep, though mentioning them as potential uses is appropriate.
- **Weakness - "Lack of Cross-Dataset Analysis" and "Insufficient Baseline Comparisons":** These are valid as suggestions for improvement (and are noted in the "Major" weakness about demonstrating unique value). However, as standalone criticisms demanding that a *dataset paper* must include extensive cross-dataset benchmarking, they are softened to a "nice-to-have" as they go beyond the core act of presenting the dataset.
- **Weakness - "Limited Technical and Methodological Novelty":** The primary contribution is the dataset itself, not a new algorithm. For a dataset paper, novelty is derived from the data's unique characteristics (eKYC videos, demographic balance, ethical collection), which are present.

## Suggestions
- **Revise Evaluation Protocol:** Replace the fixed-threshold verification metric with standard benchmarks (e.g., report TAR at various FARs, EER, or ROC curves). Consider defining a protocol that uses video sequences as probes against a reference, applying temporal pooling.
- **Temper Claims Regarding Scale:** Revise language that overstates the dataset as "comprehensive" or a definitive "new benchmark." Acknowledge the scale limitation while emphasizing its unique value for controlled studies on eKYC dynamics and demographic fairness.
- **Add a Comparative Experiment:** Include one clear experiment comparing a standard model's performance on VIBEFACE versus another dataset (e.g., SOTERIA) under a matched protocol. This would directly evidence the specific challenges your dataset introduces.
- **Expand Limitations Section:** Explicitly discuss the consequences of the 50-subject scale for statistical power and generalizability, and note the controlled studio environment versus truly "in-the-wild" data.
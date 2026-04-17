# Remotely Detectable Robot Policy Watermarking

- Decision: Accept (Poster)
- Scores: 8, 8, 6, 6

## Abstract
The success of machine learning for real-world robotic systems has created a new form of intellectual property: the trained policy. This raises a critical need for novel methods that verify ownership and detect unauthorized, possibly unsafe misuse. While watermarking is established in other domains, physical policies present a unique challenge: remote detection. Existing methods assume access to the robot’s internal state, but auditors are often limited to external observations (e.g., video footage). This “Physical Observation Gap” means the watermark must be detected from signals that are noisy, asynchronous, and filtered by unknown system dynamics. We formalize this challenge using the concept of a glimpse sequence, and introduce Colored Noise Coherency (CoNoCo), the first watermarking strategy designed for remote detection. CoNoCo embeds a spectral signal into the robot’s motions by leveraging the policy’s inherent stochasticity. To show it does not degrade performance, we prove CoNoCo preserves the marginal action distribution. Our experiments demonstrate strong, robust detection across various remote modalities—including motion capture and side-way/top-down video footage—in both simulated and real-world robot experiments. This work provides a necessary step toward protecting intellectual property in robotics, offering the first method for validating the provenance of physical policies non invasively, using purely remote observations.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces CoNoCo (Colored Noise Coherency), the first framework for remotely detectable watermarking of robot control policies. Unlike prior approaches that require white-box or onboard access, CoNoCo embeds a frequency-domain signature into the stochastic exploration noise of a policy, enabling provenance verification from remote, noisy, and asynchronous observations (e.g., video footage). The watermark is injected by shaping the policy’s Gaussian noise into a narrow spectral band and later detected via spectral coherency, which is invariant to unknown system dynamics. The method is evaluated on simulated and real-world robots (e.g., RoboMaster platform) under various sensing modalities, showing high detectability, policy utility preservation, and strong robustness to additive noise attacks. The authors also open-source their full code and trained models.

### Strengths
The paper addresses a novel and important challenge: remote watermark detection for robotic policies, representing a new research direction beyond digital or model-based watermarking. The concept of bridging the “Physical Observation Gap” through spectral coherency is both elegant and well-motivated.

The methodology is clearly described and mathematically grounded, with supporting theorems demonstrating action-distribution preservation and LTI-invariant detectability. The experiments cover a diverse set of control environments and modalities, including real-world deployment, demonstrating consistent performance.

The paper is exceptionally well-written and organized. Figures clearly illustrate the concept and pipeline. The mathematical derivations are detailed yet readable, and limitations are candidly discussed in the appendix.

The problem addressed is highly relevant to both AI safety and IP protection in robotics. Remote verification of control policies is crucial for future large-scale autonomous deployments. The release of code and trained models enhances the paper’s impact and reproducibility.

### Weaknesses
While the paper convincingly demonstrates robustness under several noise conditions, it does not discuss or evaluate how the watermark performs under unseen or novel distortions beyond the tested scenarios. Real-world observation pipelines may involve motion blur, lighting changes, camera compression, occlusions, or domain shifts in dynamic, conditions under which the spectral coherency assumption might weaken. Without empirical evidence or analysis of these unseen distortions, it remains uncertain how broadly the robustness claim generalizes in practice.

### Questions
1. How does CoNoCo perform under unseen distortions such as camera compression, lighting variation, partial occlusion, or non-linear distortions that break LTI assumptions?

2. Could frequency-domain regularization or multi-band embedding improve robustness to these distortions?

3. Does the open-source release include pretrained detectors for other robot types or example video-based detection pipelines?

4. Would combining CoNoCo with learned feature-based detectors (e.g., neural coherence estimators) further enhance robustness under non-linear observation mappings?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents an approach to encode IP protections into robot actions using colored noise coherency approaches. This adds noise to robot actions, preserving the marginal action distribution. The noise used is Colored Gaussian Noise (CGN), which replaces white noise used during exploration phase of the RL system training. Unlike dynamic watermarking methods for sensor inputs, this approach requires detectability and is not a strict defense against attackers - as such, it is also simple enough to be rapidly detectable. The paper proves that the CGN watermark detectability converges to unity as the 'glimpse sequence' of action data observed increases.

### Strengths
The paper is very theoretically sound, proving its claims in theory before moving to experimentation, in one case on real robot hardware. It is also thorough in its discussion on limitations, open questions and questions such as attack resilience of the CoNoCo approach.

### Weaknesses
For a reinforcement learning based paper that focuses on IP protections in robotics, it seems too thin on the experimental section to me, but otherwise is excellent.

### Questions
1. Could the authors comment on the complexity of robots such as multi-jointed arms that have to also conduct fine manipulation? Would CoNoCo be applicable there, given the smaller margin of error for those manipulations?

2. While the authors address the limitations of glimpse sequence sensor data quality requirements in the appendices, I would like to know if approaches involving signal reconstruction under noise would allow more robust watermarking without the need to fit the glimpse sequence length required to the periodicity of the signal (as mentioned in the periodicity part of open questions).

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes Colored Noise Coherency, or CoNoCo, a watermarking strategy for robotic control policies that can be detected from remote observations. The core idea is to replace standard white exploration noise with band limited colored noise and to detect the resulting spectral signature through coherency in the frequency domain. The authors argue that coherency is invariant to unknown linear time invariant dynamics, which makes the detector robust to the physical observation gap between actions and sensed motion. The study reports strong detection from motion capture and monocular video while preserving the marginal action distribution and task reward.

### Strengths
- Clear problem formulation of remote watermark detection with only glimpse sequences and a careful breakdown of synchronization uncertainty, dynamics, and noise.
- A principled detector based on spectral coherency that is motivated by standard results in signal processing and that aligns well with the physical setting.
- Broad experimental sweep across simulated and real platforms with multiple sensing modalities, including top down and side view video, with anonymization tests and ROC based reporting.

### Weaknesses
## 

- Watermarks are not detected in the presence of obstacles in the navigation task. It remains to see if the CoNoCo policy characteristics would be detectable in a general cluttered environment.
- Inability to Handle Time Offsets: This is a major operational weakness. The paper states that CoNoCo "does not handle large time offsets well" and that detection requires the "glimpse data recording needs to start near the beginning of the robot's operations". In any realistic scenario (like pulling CCTV footage), an auditor will be "tuning in" at an arbitrary time, not at the precise moment the robot was activated. Ideally the watermarking sequence $W_k$ would be ran in short intervals or a study of the time offset detection capabilities w.r.t. offset and detection length required would be included.
- The authors test a naive Additive Noise Attack in Appendix G. This attack involves adding White Gaussian Noise (WGN) to the policy's actions before execution but the attacker has no other objectives like maintaining performance.
    - An adversarial RL agent would not just add *random* noise. It could be trained with a multi-objective reward function:
        1. Maximize the original task reward (to maintain performance).
        2. Add random noise or change the policy in a structured way.
    This RL agent would learn to output a structured jamming signal that would most likely interfere with the spectral signature in the secret frequency band $\mathcal{B}$.
    - Another option is policy distillation [1,2] where the adversary learns to copy the behavior of the watermarked policy while maintaining performance. This could effectively change the policy and thus remove the watermark.

### References

[1] Policy Distillation, Andrei A. R. et al., 2015

[2] Refined Policy Distillation: From VLA Generalists to RL Experts, Tobias J. et al., IROS 2026

### Questions
1. In the robustness to adversarial additive noise experiments in Appendix G, is the adversary given any objective to maintain performance, for example a reward preserving constraint or penalty on deviation from nominal actions?
2. Does the detector ever raise false positives on non watermarked policies that naturally concentrate energy in the secret band due to task dynamics, and what priors or band selection rules mitigate this risk?
3. How sensitive is detection to modest drift in the policy execution rate during a single deployment, and can the search grid adapt online?
4. How is policy distillation or behavior cloning as a way to remove the watermarking detection capability?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper aims to design watermark for trained robotic policies in the observation-only setting. That is, to verify the ownership of a robot’s policy using only remote observation such as videos or motion capture. To make this concrete, the authors identifies the “Physical Observation Gap,” which captures the three key challenges. The approach focuses on Gaussian policies.

The method CoNoCo adds a covert spectral signature into the exploration noise of the policy by concentrating energy in a secret frequency band. This modification preserves the marginal distribution over actions, ensuring task performance is unaffected. The detector uses it to reconstruct and compare the signature via spectral coherency, scanning over possible execution rates.

Theory shows the watermark remains statistically invisible per timestep but can be detected over time. Experiments across real and simulated environments including RoboMaster, VMAS, and MuJoCo tasks demonstrate strong detection performance, reward preservation, and some robustness to noise.

### Strengths
1. Well-scoped and original problem: The paper clearly frames a new challenge—verifying the ownership of a robot’s policy using only remote sensing (e.g., video), with no white-box access. The proposed “Physical Observation Gap” is realistic and well-formulated, addressing timing mismatches, unknown dynamics, and sensing limitations.

2. Simple but clever method: The idea to use colored Gaussian noise with energy concentrated in a secret frequency band is elegant. It avoids changing the marginal action distribution while enabling detectability through spectral analysis. The implementation is straightforward and practical.

3. Solid theoretical grounding: The paper gives intuitive and mathematically sound analysis. It shows that marginal action distributions are preserved and that the coherence metric used for detection has a direct relationship with SINR.

4. Strong and diverse experiments: The authors validate the method in both simulation and a real robot setting, using various sensing modalities including motion capture and single-camera video. They report strong detection results, reasonable robustness, and include anonymity comparisons with a baseline.

### Weaknesses
1. Limited attack robustness: The experiments mainly test additive noise. But real-world attackers might apply frame drops, time shifts; none of which are evaluated here. These could undermine coherency-based detection.

2. Scope is restricted to continuous Gaussian policies: There’s no discussion on how this approach might extend to discrete or deterministic policies, which are common in practice

### Questions
1. Attack resilience: How does your method perform under time distortions, missing frames, or camera shifts? Any strategies to make it more invariant?

2. Beyond Gaussian policies: Do you see a way to adapt this idea to deterministic or discrete-action policies while retaining reward preservation and detectability?

### Soundness
3

### Presentation
3

### Contribution
3

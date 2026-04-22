# Keep It Real: Challenges in Attacking Compression-Based Adversarial Purification

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Previous work has suggested that preprocessing images through lossy compression can defend against adversarial perturbations, but comprehensive attack evaluations have been lacking. In this paper, we construct strong white-box and adaptive attacks against various compression models and identify a critical challenge for attackers: high realism in reconstructed images significantly increases attack difficulty. 
Through rigorous evaluation across multiple attack scenarios, we demonstrate that compression models capable of producing realistic, high-fidelity reconstructions are substantially more resistant to our attacks. In contrast, low-realism compression models can be broken. 
Our analysis reveals that this is not due to gradient masking. Rather, realistic reconstructions maintaining distributional alignment with natural images seem to offer inherent robustness. 
This work highlights a significant obstacle for future adversarial attacks and suggests that developing more effective techniques to overcome realism represents an essential challenge for comprehensive security evaluation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates the relationship between realism in image compression and adversarial robustness. It argues that realistic reconstructions produced by generative or learned compression models improve robustness against adversarial perturbations. The authors evaluate several compression-based defenses under both standard and adaptive attacks and conclude that realism, rather than compression ratio, is the key factor contributing to robustness.

### Strengths
1.The paper evaluates multiple adaptive attack settings, which provides a thorough empirical basis.

2.The paper provides some insights into the correlation between realism and robustness.

### Weaknesses
1.Although the paper convincingly demonstrates an empirical link between realism and robustness, it lacks a clear theoretical explanation of why realism leads to robustness.

2.The comparison with diffusion-based defenses is limited. The paper evaluates only under standard PGD without employing stronger adaptive attacks (such as U-Net BPDA or ARA) that could more rigorously test the claimed efficiency–robustness trade-off.

3.The discussion of realism metrics such as FID remains qualitative. The experiments do not provide quantitative analyses that directly link realism scores to robustness outcomes.

4.The paper offers limited conceptual novelty beyond existing empirical trends. Its implications for designing practical and efficient defenses are not sufficiently developed.

### Questions
Please refer to the weaknesses.

### Soundness
2

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
2

### Summary
Summary

This paper studies compression-based adversarial defenses, focusing on the role of realism in reconstructed images. The authors argue that realism—not merely compression or distortion—significantly improves robustness against adaptive attacks. They evaluate multiple learned compression models (e.g., Hyperprior, HiFiC, MRIC, CRDR) on ImageNet using diverse threat models (black-, gray-, and white-box) and adaptive attacks (BPDA, U-Net BPDA, ACM, ARA).

 The main finding is that high-realism compression models consistently exhibit stronger resistance to adversarial perturbations without relying on gradient masking. Visualization of loss landscapes and extensive ablations support the claim that realism maintains distributional alignment with natural images, thus hindering attacks.

### Strengths
Strengths

1. Comprehensive evaluation across architectures, defenses, and adaptive attack types, addressing the common critique of weak adversarial testing.

2. Novel insight: Identifies realism as a key determinant of robustness, offering a new conceptual lens beyond distortion or gradient obfuscation.

3. Well-written and reproducible: Clear methodology, detailed threat models, and code availability.

### Weaknesses
Questions for the Authors

1. Although high-realism models exhibit stronger robustness, do such realism-enhanced defense models require higher computational cost compared to low-realism ones when deployed in practice?

2. Does the realism–robustness relation hold for detection or segmentation tasks?

3. I am still curious about the underlying mechanism through which realism enhances robustness. As the paper mentions, compression models with higher realism produce reconstructions that are more consistent with the natural image distribution, which benefits the classifier’s decision-making. Another possible explanation is that high-realism models can effectively remove adversarial noise. Which of these two factors is the dominant cause? Moreover, what is the relationship between improving realism and removing adversarial perturbations?

### Questions
see weakness

### Soundness
3

### Presentation
4

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
This paper argues that realism, not low distortion, is the key to robust compression-based adversarial defenses. Through attacks, the authors demonstrate that high-realism models are significantly more resilient. The paper's primary contribution lies in its novel perspective, which shifts the focus of compression-based defenses from traditional distortion metrics to the dimension of realism for the task of image classification. This hypothesis is supported by a rigorous evaluation that employs a strong suite of attacks. The evidence from these targeted experiments helps to decouple the defensive benefits of realism from confounding factors like gradient masking artifacts, strengthening the paper's core claims within this specific task.

### Strengths
The paper's primary contribution lies in its novel perspective, which shifts the focus of compression-based defenses from traditional distortion metrics to the dimension of realism for the task of image classification. This hypothesis is supported by a rigorous evaluation that employs a strong suite of attacks. The evidence from these targeted experiments helps to decouple the defensive benefits of realism from confounding factors like gradient masking artifacts, strengthening the paper's core claims within this specific task.

### Weaknesses
1. The evaluation is confined to classification. The defense's "hallucination" mechanism may harm pixel-sensitive tasks (e.g., segmentation), limiting generalizability.
2. The paper fails to report or control for the bitrate (bpp) of the compression methods under comparison. Bitrate is a critical variable that directly impacts baseline classification accuracy and reconstruction quality.
3．The clarity of some figures is insufficient for readers unfamiliar with the sub-field. For example, the caption for Figure 1 does not explain the meaning of the stacked color shades, nor does it define terms like "4/255", forcing the reader to hunt for definitions in the main text.
4. The main paper lacks compelling qualitative visuals (like the visualizations in Figure 8).

### Questions
1.The findings are focused entirely on classification. Given that the defense's core mechanism is "hallucinating" plausible details , have the authors considered how this might impact pixel-sensitive tasks like semantic segmentation? 
2.Could the authors please provide the bitrates (bpp) for the different compression models as configured for the main experiments (e.g., JPEG, ELIC, and CRDR )? This information is critical for a fair comparison, as bitrate directly impacts baseline accuracy and could be a significant confounding variable for the robustness gains attributed to realism.

### Soundness
2

### Presentation
2

### Contribution
1

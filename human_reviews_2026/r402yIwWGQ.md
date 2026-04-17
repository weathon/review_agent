# RAID: Towards Robust AI-Generated Image Detection with Bit Reversed Images

- Decision: Reject
- Scores: 6, 4, 6, 4

## Abstract
The rapid advancement of image generation models has made it increasingly difficult for people to distinguish AI-generated images from real ones. To prevent the potential risks associated with the misuse of fake images, AI-generated image detection has gained significant attention. Existing methods neglect the inherent differences between real and fake images, thus lacking robustness and generalization ability. In this work, we innovatively investigate AI-generated image detection using bit-planes, and introduce the bit reversed image. We propose a simple yet effective pipeline consisting of construction of bit reversed images, gradient-based patch selection and a convolutional classifier. Extensive experiments on more than 32 benchmarks verify the effectiveness of our approach across different settings, including evaluations of generalization capability and zero-shot performance. Particularly, our approach achieves nearly 100% accuracy on eight benchmarks for cross-generator evaluation on the GenImage dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper proposes RAID, a novel framework for AI-generated image detection leveraging bit-plane decomposition and the introduction of Bit Reversed Images (BRI). By reversing bit-plane order, the method amplifies subtle artifacts invisible in original images. Combined with a gradient-based patch selection and a lightweight modified ResNet-50, RAID achieves strong cross-generator and zero-shot generalization across 40+ benchmarks.

### Strengths
- The idea of using bit-plane reversal for AI-generated image detection is fresh and unexplored. It introduces a simple yet conceptually interesting perspective beyond spatial and frequency domains. The method reinterprets bit-plane representation — commonly used in steganography — into a discriminative cue for forgery detection, showing creative cross-domain thinking.
Moreover, the integration with a gradient-based patch selector (GBPS) is efficient and complements the novel bit-reversal transformation.
- The experiments are extensive, covering multiple datasets (AIGCDB, GenImage, GID, GVD) and diverse setups (cross-generator, zero-shot, robustness).
- Quantitative results are strong, with clear ablations demonstrating the role of each component.
- The paper is generally well-structured, with detailed methodology and visualization of bit-planes, bit-reversed effects, and patch-level predictions.
- Figures (e.g., Fig. 1–5) are informative and aid understanding of how bit reversal exposes artifacts.
- Given the current importance of detecting synthetic images, this work contributes a lightweight, interpretable, and effective detector. Its generalization to unseen generators and real-world degradation scenarios underscores robustness and potential for deployment.

### Weaknesses
- The paper lacks a clear theoretical justification for why bit reversal amplifies AI-specific artifacts. There is no spectral or statistical evidence to show that reversing bit-plane order meaningfully highlights generative inconsistencies rather than generic noise.
- Some strong recent baselines (e.g., C2P-CLIP, NPR, FatFormer, CoD) are missing, limiting the claim of state-of-the-art performance.
- The comparison under image perturbations (Table 6) includes only ESSP, while stronger pretrained universal detectors (e.g., UnivFD) are not discussed.
- Table 4 shows certain bit-forward configurations nearly match bit-reversed performance (98.0% vs. 98.4%), raising questions about the true necessity of reversal. The paper should clarify why full reversal is superior beyond empirical coincidence.
- It remains unclear whether bit reversal corresponds to frequency inversion (i.e., swapping high- and low-frequency components). Without clarifying this, it is hard to position the method relative to existing frequency-based approaches.

### Questions
- Can the authors theoretically or empirically justify why bit reversal highlights generative artifacts? Is there a measurable spectral or statistical property supporting this observation?
- Does the bit reversal operation correspond to swapping high- and low-frequency components in the frequency domain? If not, how does it differ from such transformations?
- Table 4 shows bit-forward images can also reach 98.0% accuracy. Why, then, is full reversal necessary?
- Why are comparisons under image perturbations limited to ESSP? How does RAID perform against pretrained universal detectors such as UnivFD, CoD, or FatFormer?
- Could combining BRI with multi-scale or transformer-based architectures further enhance robustness or interpretability?
- What I care most about is what traces the bit-reversed image extracts and its relationship with frequency inversion (i.e., swapping high- and low-frequency components). I will adjust the score based on the author's response.

### Soundness
3

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
This paper proposes RAID, a robust AI-generated image detector that leverages bit-plane decomposition, specifically introducing “bit reversed images” as feature amplifiers for distinguishing real from synthetic imagery. The approach involves lossless bit-plane decomposition, strategic reordering to amplify high-frequency details, gradient-based patch selection for focusing on artifact-rich regions, and a modified ResNet-50 classifier. Extensive experiments are conducted on over 40 benchmarks (AIGCDB, GenImage, and new datasets), with comprehensive ablations and cross-generator/cross-dataset/zero-shot generalization analyses. RAID demonstrates superior performance and is computationally efficient.

### Strengths
1. The paper is among the first to successfully operationalize bit-plane analysis and bit reversal as a signal for robust artifact amplification, providing a refreshingly simple yet powerful perspective in the deepfake detection landscape.
2. The construction of bit reversed images, followed by gradient-based patch selection and adapted ResNet-50 deployment, is both theoretically sound and easy to implement. The method’s fast, non-parametric upfront steps (bit-reversal and patch selection) are especially appealing for scalability.

### Weaknesses
1. The paper omits a discussion and experimental comparison with “LOTA: Bit-Planes Guided AI-Generated Image Detection” , which is highly relevant, as it also explores bit-plane signals for AI-image detection. The absence of this reference undermines the claim of being among the first to leverage bit-planes and makes it harder to gauge the true incremental value over related work. This missing context should be addressed both in the Related Work section and, ideally, with an experimental baseline.
2. While the intuition for artifact amplification via bit reversal is plausible and empirical results are strong, Section 3.1 could benefit from a deeper mathematical analysis or visualization of why and how bit reversal leads to more pronounced differences in the artifacts between real and AI-generated images. For instance, it would be useful to connect this operation more formally to the types of noise and structure generated by different models.
3. Although RAID is benchmarked against several prior patch-based and frequency/spatial methods, comparison to newer or alternative handcrafted artifact detectors—such as those utilizing global color analysis, learned mask regularization, or ensemble patch aggregation—appears limited. This is especially pertinent for Table 1 and Table 2, which emphasize detection efficacy and generalization, and could be expanded with additional strong baselines.
4. For zero-shot and cross-dataset generalization, relying only on ImageNet real images may not be ideal, as ImageNet’s distribution biases could influence results. Consideration of multiple, diverse “real” datasets would strengthen generalization claims.

### Questions
1. Could the authors provide a more formal or statistical explanation (beyond visual intuition) for why bit reversal amplifies artifacts in AI-generated images more than in real ones? Are there measurable frequency or entropy-based differences that could make this hypothesis falsifiable?
2. Why was “LOTA” not covered or compared directly, given its direct relevance? Would including a LOTA baseline or at least a detailed discussion alter RAID’s claimed superiority or novelty?
3. Are there any scenarios (dataset types, generative models, manipulations) where RAID demonstrably fails or produces high rates of false positives/negatives—and what are the typical characteristics of such cases?
4. In Table 10, what constraints (if any) govern the selection/learning of bit-plane weights? Is there a principled pathway to learn these weights beyond brute-forcing variants, or to adapt them to novel distributions?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes RAID, a novel and efficient method for detecting AI-generated images. This method reverses the bit-planes of images as a preprocessing step, and uses a gradient-based patch selection method to find the optimum patch from the bit-reversed image, which is passed to a CNN classifier for detecting AI-generated images. The authors validated their result on AIGCDB and GenImage benchmarks.

### Strengths
The key strength of the paper is the creative application of bit-planes to AI-generated image detection. This method achieves a significantly improved score in the GenImage and AIGCDB benchmark and demonstrates superior generalization on unseen generators. The authors demonstrated that this method is highly efficient as well. Lastly, Table 4 provided interesting insight into the effect of bit reversal and bit forwarding in different generators.

### Weaknesses
While the empirical results are significant, the paper lacks a deeper analysis of the effectiveness of Bit Reversed Images. The difference between the real and generated BRI images, as shown in Figure 1, needs to be more clearly demonstrated, as the provided samples do not provide a clearer picture of the difference between them.

The BRI is a novel method for highlighting high-frequency information, so a detailed discussion of why this specific transformation is capable of outperforming existing methods that utilize frequency domain information would have clarified the unique advantages of the paper.

The results, especially in Table 4, suggest that all the existing generators share a fundamental flaw in the statistical properties of the pixel-level noise, and this can be modeled using bit-reversed images. This generalization capability of BRI is highly interesting. But the paper did not explain or analyze its powerful implications more explicitly, which reduces its full impact.

### Questions
1. The paper’s motivation is dependent on the claim that BRIs for generated images contain noticeable artifacts. Could the authors provide additional analysis to support this visual claim?
2. Could the authors elaborate on why this specific transformation is more effective than prior frequency domain methods?

### Soundness
2

### Presentation
2

### Contribution
4

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper proposes **RAID** (“Robust AI-generated Image Detection”), a simple yet seemingly effective approach for detecting AI-generated images. The method converts each image into a *bit-reversed version*, where the least significant bits (LSBs) are assigned the highest intensity weights and vice versa, supposedly to highlight subtle generation artifacts. A gradient-based patch selector then chooses the region with the strongest signal, and a lightweight CNN performs classification. On multiple cross-generator benchmarks (e.g., GenImage), the method reports **>98% accuracy**, outperforming heavier baselines.

### Strengths
* The idea is **novel in presentation** — flipping bit-plane importance to emphasize low-level noise patterns is clever and computationally light.
* The results, if reproducible, are **remarkably strong**, showing high cross-generator generalization, suggesting that the authors tapped into a genuinely discriminative low-level signal.
* The ablation studies and bit-order experiments are thorough; the performance gains for the full bit-reversal and patch-selection combination are consistent.

### Weaknesses
1. **Theoretical basis is weak.**
   The paper speculates that diffusion or GAN generators fail to reproduce LSB-level noise distribution, and that reversing bit weights amplifies this difference. But there’s **no evidence or measurement** of these bit-plane distribution gaps. It’s a plausible but unverified hypothesis.

2. **Amplifying features ≠ discovering new signal.**
   Simply amplifying certain bit planes (like a hand-crafted high-pass filter) doesn’t guarantee generalization. A CNN or Transformer could, in principle, learn to emphasize LSBs or other frequency bands by itself. The gains might stem from shortcut cues or dataset quirks rather than truly meaningful LSB statistics.

3. **Results feel too good for such a simple trick.**
   The jump to ~98% accuracy across unseen generators using only a linear remapping plus a small CNN seems **disproportionately large**. Given the simplicity of the transformation, it’s unclear how such a method achieves such robust cross-distribution generalization without overfitting to codec or pipeline artifacts.

4. **Lack of code, demo, or reproducibility.**
   The paper doesn’t release code or pretrained models, making it impossible to verify the findings. Without replication, the very high reported numbers remain speculative.

5. **Practical robustness missing.**
   The method degrades sharply under realistic conditions — e.g., JPEG-90 or Gaussian blur reduce accuracy to ~75%. Social media and messaging platforms re-compress images heavily, so a real-world detector must survive such transformations.

### Questions
1. **Detector vs. LSB patterns:**
   You argue that standard CNN-based detectors cannot effectively learn small LSB-level differences, hence the need for bit-reversal. Can you provide **proof or analysis** supporting this claim?

2. **Reproducibility and transparency:**
   The reported results are exceptionally high for such a lightweight approach. Can you please **release code, pretrained weights, or a simple demo** so reviewers and researchers can test your method on *unseen images*?

### Soundness
1

### Presentation
2

### Contribution
3

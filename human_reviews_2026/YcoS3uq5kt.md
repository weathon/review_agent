# GuardAlign: Test-time Safety Alignment in Multimodal Large Language Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 4, 4

## Abstract
Large vision-language models (LVLMs) have achieved remarkable progress in vision–language reasoning tasks, yet ensuring their safety remains a critical challenge. Recent input-side defenses detect unsafe images with CLIP and prepend safety prefixes to prompts, but they still suffer from inaccurate detection in complex scenes and unstable safety signals during decoding. To address these issues, we propose **GuardAlign**, a training-free defense framework that integrates two strategies. First, OT-enhanced safety detection leverages optimal transport to measure distribution distances between image patches and unsafe semantics, enabling accurate identification of malicious regions without additional computational cost. Second, cross-modal attentive calibration strengthens the influence of safety prefixes by adaptively reallocating attention across layers, ensuring that safety signals remain consistently activated throughout generation. Extensive evaluations on six representative MLLMs demonstrate that GuardAlign reduces unsafe response rates by up to 39\% on SPA-VL, while preserving utility, achieving an improvement on VQAv2 from 78.51\% to 79.21\%.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper focuses on safety alignment in multi-modal large language models (mllms). Recent studies reveal that input-side defenses detect unsafe images with CLIP and prepend safety prefixes to prompts, but most MLLMs suffer from inaccurate detection in complex scenes and unstable safety signals during decoding. To this end, the authors propose GuardAlign, which is a training-free defense framework that first leverages optimal transport to measure the distribution distances between image patches and unsafe semantics, then cross-modal attentive calibration ensures the safety signals remain consistently activated throughout generation. Extensive evaluations on six representative MLLMs demonstrate that GuardAlign reduces unsafe response rates by up to 39% on SPA-VL.

### Strengths
1.	This paper starts from an interesting motivation. In order to ensure the MLLMs are not guided by the dangerous information carried in the images, the authors come up with a method to mask the dangerous regions out of the images.

2.	The proposed method GuardAlign makes sense to me. To achieve the above goal, the authors propose OT-enhanced safety detection module, which first divides the image into patches, then measures the similarity between the patches and potential harmful content.

### Weaknesses
My main concern is about the negative effect that could be brought by the GuardAlign. Even though Table 6 demonstrates that their method preserves utility and yields consistent gains, the reasons behind are not revealed and discussed enough. Intuitively, masking the image leads to missing information about the image, but the performance is not affected. Is it because the masking region is small or something? If so, relevant analysis should be performed.

### Questions
Please refer to Weaknesses No.1.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes GuardAlign, a training-free defense framework for enhancing the safety of MLLMs. The method integrates two key strategies: OT-enhanced safety detection and cross-modal attentive calibration. Extensive evaluations on six MLLMs demonstrate that GuardAlign significantly reduces unsafe response rates while preserving the utility of MLLMs. The approach is efficient, adding minimal inference overhead compared to normal inference.

### Strengths
- Training-free efficiency: GuardAlign operates entirely at inference time without requiring additional data or fine-tuning, making it highly practical and resource-efficient. 

- Comprehensive experimental validation: The paper provides thorough evaluations across multiple safety benchmarks and utility tasks, including detailed ablation studies and efficiency analyses. 

- Low inference overhead: Compared to existing inference-time defenses like ETA, GuardAlign achieves better safety with moderate computational cost, striking a balance between effectiveness and efficiency.

### Weaknesses
- Utility improvement: While the paper reports that GuardAlign avoids performance degradation and even boosts utility (e.g., VQAv2 accuracy improves from 78.51% to 79.21%), the underlying mechanism is not sufficiently analyzed. It remains unclear why masking unsafe patches or calibrating attention would enhance general capabilities—this warrants further theoretical or empirical justification.  

- Limited model scale evaluation: Experiments are confined to MLLMs up to 13B parameters (e.g., LLaVA-1.5-13B). Given the trend toward larger models (e.g., 70B+), validating GuardAlign on more scalable architectures would strengthen its generalizability and impact.  

- Practicality concerns: The cross-modal attention calibration requires direct manipulation of attention maps within the MLLM backbone, which may involve accessing and altering internal model structures. This could limit applicability in black-box or proprietary systems where such modifications are restricted, reducing the method's versatility in real-world scenarios.

### Questions
Please see the weakness part.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper proposes GuardAlign, a training-free framework to enhance the safety of multimodal large language models (MLLMs). It integrates two strategies: (1) OT-enhanced safety detection using optimal transport to accurately identify unsafe image regions, and (2) cross-modal attention calibration that reinforces safety prefix signals during text generation. Experiments on six major MLLMs demonstrate that GuardAlign significantly reduces unsafe response rates while maintaining or slightly improving task performance, achieving up to 39% reduction in unsafe outputs without retraining or additional data.

### Strengths
The paper is original in combining optimal transport-based detection with attention calibration for inference-time safety alignment. 

The technical quality is solid, with rigorous theoretical analysis and comprehensive evaluations across models and datasets. 

Clarity is high. both intuition and formulation are clearly articulated, and experimental design is systematic.

### Weaknesses
The method, while efficient, introduces several hyperparameters (e.g., τ, γ) that are not fully analyzed for stability or generalizability. 

Evaluation is limited to vision–language reasoning; applicability to other modalities remains untested. 

The detection component depends on CLIP backbones, which could inherit existing biases.

### Questions
How sensitive is GuardAlign's performance to its threshold and amplification parameters across unseen datasets? 

Could the OT-based detection misclassify benign but semantically rich regions (false positives)? 

Would combining GuardAlign with post-hoc fine-tuning methods yield additive benefits?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes GuardAlign, a training-free, inference-time defense for multimodal LLMs (MLLMs) that combines two complementary modules:

OT-enhanced safety detection — use patch-level Optimal Transport (OT) between image patch features and a set of predefined “unsafe” prompt embeddings to identify and mask visual regions that align with harmful semantics;

Cross-modal attention calibration — amplify attention toward a prepended safety prefix across selected middle fusion layers so that the safety signal remains activated during autoregressive decoding.

### Strengths
+ GuardAlign operates at inference time and requires no fine-tuning, which is attractive for rapid deployment on large deployed MLLMs.
+ The coupling of fine-grained OT-based patch scoring with attention-level calibration for safety prefixes is an intuitive and novel pairing: detect & sanitize visual evidence, then ensure the LLM heeds the safety prefix.

### Weaknesses
- The method is evaluated on many benchmarks but primarily in a black-box or benchmarked adversary setting. An adaptive attacker that crafts images to both avoid OT detection and trigger unsafe generations (e.g., by distributing harmful signals over many patches or embedding signals in texture) is not evaluated. GuardAlign’s resilience to adaptive/strong adversaries is unclear.

### Questions
refer to weakness

### Soundness
3

### Presentation
3

### Contribution
3

# Exposing Hallucinations To Suppress Them: VLMs Representation Editing With Generative Anchors

- Decision: Reject
- Scores: 2, 6, 6, 2

## Abstract
Multimodal large language models (MLLMs) have achieved remarkable success across diverse vision-language tasks, yet they remain highly susceptible to hallucinations, producing content that is fluent but inconsistent with visual evidence. Such hallucinations, spanning objects, attributes, and relations, persist even in larger models, while existing mitigation approaches often require additional finetuning, handcrafted priors, or trade-offs that compromise informativeness and scalability. To address this limitation, we propose a training-free, self-supervised method for hallucination mitigation. Our approach introduces a novel hallucination amplification mechanism: a caption is projected into the visual space via a text-to-image model to reveal implicit hallucination signals, serving as a negative anchor, while the original image provides a positive anchor. Leveraging these dual anchors, we edit decoder hidden states by pulling representations toward faithful semantics and pushing them away from hallucination directions. This correction requires no human priors or additional training costs, ensuring both effectiveness and efficiency. Extensive experiments across multiple benchmarks show that our method significantly reduces hallucinations at the object, attribute, and relation levels while largely preserving recall and caption richness, e.g., achieving a hallucination  reduction by over 5% using LLaVA-v1.5-7B on CHAIR. Furthermore, results on diverse architectures, including LLaVA-NEXT-7B, Cambrian-8B, and InstructBLIP-7B, validate strong cross-architecture generalization. More importantly, when applied to hallucination-free captions, our method introduces almost no side effects, underscoring its robustness and practical plug-and-play applicability. The implementation will be publicly available.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper introduces a training-free hallucination mitigation approach for MLLMs.
The proposed method works as follows: First, a caption is generated from the MLLM using the original image. Next, the caption and the original image are used to generate a new image. Then, the differences between the original and synthetic image representations are used to guide the QA tasks. This idea is tested on four MLLMs, namely LLaVA 1.5, InstructBLIP, LLaVA-Next, and Cambrian, and evaluated on CHAIR, POPE, and MME.

### Strengths
- The proposed idea of generating an image from the MLLM-generated caption and the original image, and then using the differences in their latent visual representations to mitigate hallucination, seems interesting.
- The paper is well-written and easy to follow.
- This paper addresses object hallucination, which remains a critical problem in MLLMs.

### Weaknesses
* The proposed method does not appear to be an efficient, scalable, or general solution:

  * Image generation is a computationally expensive process.
  * While it may work for simple images containing few objects (e.g., around 5-10 objects), it is likely to fail in complex scenes with multiple objects (e.g., 30+ objects).

* Given this approach requires two auxiliary tasks—caption generation and image generation—before producing an answer. This additional cost should be justified by a significant performance gain. The authors do not show any specific study to report the computation overhead and performance gain; Nevertheless the reported gain is very minimal, making it a unsuitable choice in real-world.

* The experimental setup is weak, as it is evaluated only on POPE and CHAIR, which are outdated and do not comprehensively assess hallucination in LVLMs. POPE evaluates object existence on just 500 images and does not account for other forms of hallucination, such as those involving object attributes or relations. Similarly, CHAIR evaluates only 500 MSCOCO images with limited ground-truth information. These benchmarks lack diversity and rigor. The authors are encouraged to use more recent and challenging benchmarks such as HallusionBench, AMBER, M-HalDetect, and GAVIE.

* The paper relies on older MLLMs such as InstructBLIP and LLaVA 1.5, showing only marginal improvements in certain settings, which makes the effectiveness of the proposed method less convincing. To better demonstrate the method’s generality, the authors could evaluate it on more recent models such as Qwen-VL 2.5 or 3, InternVL 3, and BLIP-3. Additionally, several results (e.g., on POPE and MME) are missing for newer models like Cambrian and LLaVA-Next, which are already mentioned in the paper.

* The comparison setup is also weak, the proposed method is only compared against a few old decoding techniques such as VCD and OPERA; I recomend authors to also compared against newer training-free methods, as well as other online/offline RL methods.

### Questions
Please see the weaknesses above.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes a training-free, self-supervised method to mitigate hallucinations in multimodal large language models. The approach first reconstructs an image from a model-generated caption via a text-to-image model, exposing implicit hallucinations as explicit visual artifacts. It then forms dual visual anchors, the original image as a positive semantic reference and the reconstructed one as a negative signal, to steer the decoder’s hidden states through lightweight latent editing. Extensive experiments across CHAIR, POPE, and MME benchmarks demonstrate consistent hallucination reduction with minimal loss of informativeness and strong cross-architecture generalization. Overall, the method provides a straightforward yet effective plug-and-play solution for enhancing factual grounding in large vision-language models.

### Strengths
- The idea of projecting captions into the visual domain to amplify hallucinations is conceptually clear, technically lightweight, and eliminates the need for external labels or detectors.
- Pulling representations toward faithful semantics (+f(I)) while pushing away from hallucination directions (−f(I′)) provides a well-motivated contrastive structure that cleanly maps to latent geometry.
- The approach introduces no additional training cost and integrates smoothly into inference pipelines, making it genuinely deployable in real systems.
- The study carefully dissects α/β weights, supervision signals, and layer depth, providing transparent insight into how each factor shapes hallucination suppression versus recall trade-offs.

### Weaknesses
- The success of hallucination amplification hinges on the generative quality of the chosen T2I model; biases or artifacts in reconstruction could introduce spurious “negative” directions.
- While intuitively appealing, the paper lacks a formal justification or analysis of why linear editing in hidden space reliably aligns with semantic hallucination dimensions.
- Since the negative anchor is derived from the model’s own (possibly erroneous) caption, the supervision may reinforce existing language-model biases rather than objectively correct them.
- Reported improvements are modest and could fall within evaluation noise; statistical significance or variance analysis is missing.

### Questions
- How sensitive is the method to the specific T2I model used—would weaker backbones still expose hallucinations effectively?
- Does applying latent editing at multiple decoder layers cumulatively improve results, or does interference occur between layers?
- Did you conduct any statistical significance tests to verify the robustness of your experimental results?

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
4

### Summary
This paper proposes a training-free, self-supervised method to mitigate hallucinations in multimodal large language models. It leverages a hallucination amplification mechanism, using text-to-image projections as negative anchors and real images as positive anchors to correct decoder representations. The approach significantly reduces object-, attribute-, and relation-level hallucinations without finetuning or performance loss. Experiments show strong cross-architecture generalization and minimal side effects on faithful captions.

### Strengths
1. The proposed training-free, self-supervised framework effectively mitigates hallucinations in MLLMs without additional finetuning or handcrafted priors, offering an efficient and scalable low-cost solution across object, attribute, and relation levels.
2. The innovative hallucination amplification mechanism with dual visual-text anchors enables fine-grained detection and correction of subtle hallucinations while preserving informativeness and maintaining model reliability.

### Weaknesses
1. The paper claims to suppress three types of hallucinations: "object-level, attribute-level, and relation-level". However, experiments are only conducted on CHAIR (object-level), POPE (object-level), and MME (attribute-level) benchmarks, with no dedicated evaluation designed for relation-level hallucinations. The benchmark in [a] should be considered.
2. The paper only compares with a limited number of inference-stage methods, such as OPERA, ICD, and VCD, and does not incorporate mainstream training-stage methods (e.g., hallucination suppression based on RLHF, visual supervision fine-tuning) for comparison. Methods such as PerturboLLaVA [a] and RLAIF-V [b] should be added. 
3. The paper fails to clarify the dimensional matching and semantic alignment of the original image embedding (positive anchor) and the reconstructed image embedding. For example, given visual content differences between the original and reconstructed images (e.g., the latter contains hallucinated objects), it remains unaddressed whether their embeddings—generated via the same image encoder and projection head—reside in the same semantic space. If spatial misalignment exists, the validity of the adversarial correction logic ("pulling toward the positive anchor and pushing away from the negative anchor") is also unconfirmed, with no relevant verification or theoretical explanation provided.


[a] Chen, Cong, et al. "PerturboLLaVA: Reducing multimodal hallucinations with perturbative visual training." arXiv preprint arXiv:2503.06486 (2025).
[b] Yu, Tianyu, et al. "Rlaif-v: Open-source ai feedback leads to super gpt-4v trustworthiness." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
See the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates the vulnerability of large vision-language models (LVLMs) to hallucinations triggered by visual perturbations. The authors propose a novel visual adversarial attack framework that subtly modifies images to expose and amplify hallucination behaviors without altering the overall semantics or human perception of the image. Through experiments on several state-of-the-art LVLMs, the study shows that even small, imperceptible visual perturbations can cause severe hallucinations in both captioning and VQA tasks, revealing inconsistencies between visual grounding and textual reasoning. The method provides a systematic way to evaluate the robustness and faithfulness of LVLMs, highlighting the gap between human-aligned perception and model interpretation.

### Strengths
* The paper proposes a training-free contrastive decoding method that leverages a text-to-image (T2I) model to mitigate hallucinations in large vision-language models (LVLMs).

* Experimental results demonstrate that the proposed method effectively reduces hallucinations without compromising the general performance of the LVLM, highlighting its practicality as a lightweight, training-free solution.

### Weaknesses
* Limited novelty and missing reference. The paper omits a highly relevant prior work, ConVis (Park et al., 2025), which also employs reconstructed images from hallucinated captions to generate negative anchors and performs training-free contrastive decoding. The overlap in both motivation and method design raises concerns about the novelty of the proposed approach. The authors should include ConVis as a key reference and conduct comparative experiments to clearly differentiate their contribution.

* Dependence on the quality of the image generation model. Since the proposed method relies heavily on the T2I model, its performance may vary depending on the model’s alignment and reconstruction quality. If the generated image introduces irrelevant visual cues, the contrastive signal may fail to reduce hallucination effectively. Moreover, the current experiments appear to use a single image generation, which could be influenced by stochastic variation. Running the T2I process multiple times and reporting averaged results would strengthen the reliability of the findings.

* To further validate the effectiveness of the reconstructed image as a negative anchor, it would be beneficial to evaluate the method on additional benchmarks beyond CHAIR and POPE — for example, PhD (Liu et al., 2025) for counter-commonsense reasoning or another benchmark that assesses general performance besides MME.

Minor: The paper inconsistently uses the terms LVLM and MLLM. It would be better to standardize terminology throughout the text.

**References**:
- Park, Yeji, et al. "Convis: Contrastive decoding with hallucination visualization for mitigating hallucinations in multimodal large language models." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 6. 2025.
- Liu, Jiazhen, et al. "PhD: A ChatGPT-Prompted Visual Hallucination Evaluation Dataset." Proceedings of the Computer Vision and Pattern Recognition Conference. 2025.

### Questions
* Have the authors analyzed the latency overhead introduced by using the T2I model during decoding? A discussion or measurement of this additional computational cost would be valuable.

### Soundness
2

### Presentation
2

### Contribution
1

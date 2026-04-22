# HueManity: Probing Fine-Grained Visual Perception in MLLMs

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 4, 2, 4

## Abstract
Recent Multimodal Large Language Models (MLLMs) demonstrate strong high-level visual reasoning on tasks such as visual question answering and image captioning. Yet existing benchmarks largely overlook their ability to capture fine-grained perceptual details. As MLLMs are increasingly deployed in safety and reliability critical settings, perceptual acuity becomes essential. We present HueManity, a scalable automated benchmark for assessing fine-grained visual perception in MLLMs. HueManity comprises 83,850 Ishihara-style images embedding alphanumeric strings, designed to evaluate pattern recognition, a core aspect of visual understanding. Our evaluation of nine state-of-the-art MLLMs uncovers a striking performance deficit: the strongest model achieved only 33.6% accuracy on a simple numeric task and 3% on a harder alphanumeric task, compared to near-ceiling performance from humans (99.38%, 93.25%) and a fine-tuned ResNet-50 (96.5%, 94.5%). These findings expose a critical weakness in MLLMs’ perceptual grounding, one that remains obscured by conventional benchmarks emphasizing high-level semantics.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a synthetic visual text-reading benchmark to study how well MLLMs can perceive alphanumeric patterns in an uncommon visual representation. Specifically, the proposed benchmark represents two-letter texts in an image as a set of circles of one color against a background filled by circles of another color, known as an Ishihara Pattern. The paper shows that several popular MLLMs perform poorly in reading the text from such images, whereas human and ResNet classifiers perform almost perfectly.

### Strengths
The paper is well-written and easy to follow. The benchmark is novel and does reflect a striking limitation in MLLMs’ visual perception, which pushes against the misconception that MLLMs’ can outperform humans in all simple visual tasks. The paper also considered several MLLMs, both commercial and open-source, which increases its value as a benchmark for future MLLM development.

### Weaknesses
1. The paper misses several related papers that study the ability of MLLMs on perceiving visual details [1, 2, 3, 4], and thus does not properly place its findings in the context of other existing evidence to clarify novelty and relevance.

2. Text recognition datasets (eg, TextVQA) measure the same capability that this paper tries to measure: how well can MLLMs read text in various visual settings. Given that TextVQA contains extensive variations of text and background in real world settings, it can provide a more reliable measure of MLLMs’ overall text reading capability compared to synthetic data which may cause distribution shift. This makes the practical utility of the proposed benchmark a bit unclear: if a model L1 outperforms another model L2 on HueManity, what does it mean for real-world applications? Consider this in comparison to TextVQA that has a clear connection to real-world applications.

3. The paper only considers a single prompt and does not explore the effect of prompt variations on the performance. For example, does including instructions such as “There is a letter on an Ishihara Pattern in the image…” and/or removing the exclusion instructions, help improve the performance? This is important because MLLMs performance is quite sensitive to the prompt.

4. The paper does not explore the causes for the discovered difficulty. The mentioned potential causes in Lines 348-359 are speculations without any evidence. Quantitatively exploring some of these speculative causes can strengthen the paper’s contributions.

5. The results of MLLM fine-tuning seems to contradict the ResNet training performance. The paper does point out that it is very surprising that fine-tuning MLLMs on the task does not result in improvements, but does not explore this surprising observation further. For example, is this because the LLM is also finetuned instead of just its vision encoder? Is this because of bad choices of hyperparameters when fine-tuning? There are many missing details here that make the results unreliable. It is also unclear whether this is just a problem with Gemma, or the same applies to other MLLMs.

6. Providing quantitative results for the “MLLM Failure Patterns” could substantiate the claims in lines-425-439.

7. In Tables 2-5, Wilson intervals seem incorrect since they should not be symmetric and fall outside of [0,1]. Reporting the actual confidence interval bounds instead of +- will clarify this.

[1] MLLMs Know Where to Look: Training-Free Perception of Small Visual Details with Multimodal LLMs. ICLR 2025.

[2] Understanding Depth and Height Perception in Large Visual-Language Models. CVPR 2025.

[3] V*: Guided Visual Search as A Core Mechanism in Multimodal LLMs. CVPR 2024.

[4] Exploring Perceptual Limitation of Multimodal Large Language Models. 2024.

### Questions
1. Can the authors explain/clarify why fine-tuning the MLLM does not improve its performance? This seems contradictory to a lot of prior research and the fine-tuning results of ResNest. Also, does this happen to other MLLMs besides Gemma?

2. Can different prompts (eg, explicitly mentioning the Ishihara Pattern) change the performance?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This work proposes HueManity, a new benchmark designed to evaluate the fine-grained visual perception capabilities of multimodal large language models (MLLMs). The benchmark contains 83,850 samples, each consisting of a "blind-test" image (similar to Ishihara plates) and a corresponding question that requires the model to recognize embedded letters or numbers. This task is highly accessible for humans, achieving over 99% accuracy, and is also easily handled by lightweight models such as fine-tuned ResNet-50. In contrast, state-of-the-art MLLMs struggle significantly on this task, revealing a notable gap in their ability to perform precise, low-level visual perception—even for seemingly simple recognition tasks.

### Strengths
* This paper identifies a critical deficiency in modern MLLMs: their surprisingly weak performance in fine-grained visual perception, despite strong performance on higher-level vision-language tasks.

* The proposed benchmark, HueManity, is well-designed and presents a valuable resource for the community. It can be widely used in future work to evaluate and diagnose the fine-grained visual understanding capabilities of MLLMs.

* The authors conduct comprehensive experiments demonstrating that even state-of-the-art MLLMs struggle significantly on this task, highlighting the challenge of achieving robust, low-level perceptual accuracy in current multimodal models.

### Weaknesses
* This work focuses on a single aspect of visual understanding—recognizing characters in color-patterned images—which is relatively narrow compared to existing MLLM benchmarks. Modern benchmarks typically evaluate multiple capabilities, including low-level perception, high-level reasoning, OCR, and knowledge integration. While this task presents a challenging variant of OCR, the scope of the benchmark is limited in covering the broader spectrum of multimodal understanding expected from MLLMs.

* The benchmark appears to have high redundancy. Given the simple and repetitive nature of the task—overlaying letters and numbers on textured or colorful backgrounds—it is questionable whether 83,850 samples are necessary to reliably evaluate current MLLMs. A much smaller set (e.g., a few thousand examples) might suffice for stable assessment, especially considering that the variation is primarily in color and noise patterns rather than semantic complexity.

* While framed as an MLLM capability, the core task primarily tests the vision encoder’s ability to extract fine-grained visual features under visual noise. The language component is minimal (simple recognition questions), suggesting that the bottleneck lies in the visual encoder rather than the multimodal reasoning or language generation pipeline. Therefore, the focus may be better positioned as evaluating the fine-grained perception capabilities of vision encoders, rather than MLLMs as a whole.

### Questions
Please see weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces HueManity, a new benchmark for assessing fine-grained visual perception in Multimodal Large Language Models (MLLMs). It contains 83,850 Ishihara-style images designed to evaluate the fine-grained perceptual ability of MLLMs. The benchmark embeds alphanumeric characters within colored dot patterns to test whether models can recognize subtle visual patterns. The evaluation reveals that even top-performing models such as GPT-4.1, Claude 3.7 Sonnet, Qwen-VL Max, LLaVA-v1.6, and Pixtral perform poorly compared to human participants and a fine-tuned ResNet-50 baseline. The authors claim that these results expose a critical weakness in the perceptual grounding of MLLMs.

### Strengths
1.	The paper is well written and clearly structured, making it easy to follow.
2.	It evaluates several state-of-the-art MLLMs, including GPT-4.1, Claude 3.7 Sonnet, Qwen-VL Max, LLaVA-v1.6, and Pixtral, across two tasks: the Number Recognition Task and the Alphanumeric Recognition Task.
3.	The work provides a comparative analysis with existing MLLM benchmarks. However, some key benchmarks (e.g., MMVP [1],  MERLIM [2] and MME [3]) are missing from the evaluation.

[1] Eyes Wide Shut? Exploring the Visual Shortcomings of Multimodal LLMs (Tong et al., CVPR 2024) 
[2] MERLIM: Multi Modal Evaluation Benchmark for IT LVLMs (Villa et al., CVPRW 2025) 
[3] MME: A Comprehensive Evaluation Benchmark for Multimodal Large Language Models (Fu et al., 2023).

### Weaknesses
1.	The paper mainly reports a failure case of existing models but offers no new theoretical insights. Prior work such as Eyes Wide Shut [1] and MERLIM [2] has already shown that the visual backbones of MLLMs fail to capture fine-grained visual details.
2.	HueManity measures only color-based figure–ground discrimination under a single visual structure (Ishihara-style dots). While the idea is well motivated, it represents only a narrow and somewhat artificial subset of visual examples for evaluating  fine-grained visual perception. Other state-of-the-art benchmarks (e.g., MMVP [1], MERLIM [2], MME [3] with its OCR tasks) address this challenge from a more realistic perspective.
3.	The LoRA fine-tuning on Gemma-3-4B using only 500 samples and 3 epochs is too limited to support the claim that the issue is unlearnable. The results likely stem from optimization instability (overfitting), data non-representativeness, or implementation issues rather than a fundamental perceptual incapacity.
4.	Although the paper lists three contributions, the third one overlaps with and is effectively part of the first.

### Questions
1.	You claim that fine-grained perception is crucial for MLLMs, but is recognizing Ishihara-style digits truly representative of real-world perceptual challenges? What new insights does this benchmark provide beyond state-of-the-art alternatives such as MMVP [1], MERLIM [2], or MME [3] (which already include OCR-style tasks)?
2.	How did you ensure that the alphanumeric strings are balanced across color pairs and not biased by particular hues or contrast levels?

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
The paper uncovers a striking failure mode: mainstream MLLMs struggle to classify Ishihara-style images—even though they excel on general vision benchmarks—whereas humans and a fine-tuned ResNet-50 perform almost flawlessly. To quantify and mitigate this gap, the authors automatically generate a large-scale Ishihara-style dataset (83 850 images) and benchmark leading MLLMs. They hope the corpus will serve as a safety probe and spur progress on “out-of-distribution” visual reasoning.

### Strengths
This paper reveals an intriguing blind spot of current MLLMs and highlights a new axis for robustness research; the observation may catalyze broader studies on rare or specially structured imagery.

### Weaknesses
This paper does not investigate whether lightweight fine-tuning (rather than mere in-context learning) can already lift MLLM accuracy to near-human levels. If the deficit can be erased with a few gradient steps, the issue—and the accompanying dataset—may merit only limited attention.

### Questions
No

### Soundness
3

### Presentation
3

### Contribution
2

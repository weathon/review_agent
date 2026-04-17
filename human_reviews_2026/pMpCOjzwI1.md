# ThinkOmni: Lifting Textual Reasoning to Omni-modal Scenarios via Guidance Decoding

- Decision: Accept (Poster)
- Scores: 6, 6, 6, 6

## Abstract
Omni-modal reasoning is essential for intelligent systems to understand and draw inferences from diverse data sources. While existing omni-modal large language models (OLLM) excel at perceiving diverse modalities, they lack the complex reasoning abilities of recent large reasoning models (LRM). However, enhancing the reasoning ability of OLLMs through additional training presents significant challenges, including the need for high-quality data, task-specific adaptation, and substantial computational costs. To address these limitations, we propose ThinkOmni, a training-free and data-free framework that lifts textual reasoning to omni-modal scenarios. ThinkOmni introduces two key components: 1) LRM-as-a-Guide, which leverages off-the-shelf LRMs to guide the OLLM decoding process; 2) Stepwise Contrastive Scaling, which adaptively balances perception and reasoning signals without manual hyperparameter tuning. Experiments on six multi-modal reasoning benchmarks demonstrate that ThinkOmni consistently delivers performance improvements, with main results achieving 70.2 on MathVista and 75.5 on MMAU. Overall, ThinkOmni offers a flexible and generalizable solution for omni-modal reasoning and provides new insights into the generalization and application of reasoning capabilities.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces ThinkOmni, a training-free framework designed to enhance the reasoning capabilities of Omni-modal Large Language Models by leveraging the strengths of text-based Large Reasoning Models (LRMs). The core problem is that while OLLMs excel at perceiving diverse inputs , they lack the advanced reasoning skills of specialized LRMs. Bridging this gap usually requires costly fine-tuning or reinforcement learning, demanding large datasets and significant computation. It doesn't need training by using an LRM to guide the OLLM's decoding process at inference time. Experiments show ThinkOmni significantly boosts the performance of OLLMs on challenging multimodal reasoning benchmarks.

### Strengths
1. It avoids the need for expensive data collection, annotation, and computationally intensive fine-tuning or RL, because of its training-free nature.

2. it effectively combines the perceptual strengths of OLLMs with the reasoning strengths of state-of-the-art LRMs in a modular way.

3. it demonstrates substantial performance gains on difficult benchmarks, showing the practical effectiveness of the approach.

### Weaknesses
1. running both an OLLM and an LRM during inference will inevitably increase computational cost and latency compared to running a single model, even if optimized. 

2. the framework's performance ceiling is inherently tied to the capabilities of the chosen LRM. Errors or limitations in the LRM's reasoning will directly impact the guided OLLM.

3. the LRM guide operates on a textual representation of the omni-modal input. Complex non-textual information (e.g., intricate spatial relationships in an image, temporal dynamics in a video, nuances in audio) might be lost or distorted in this conversion, limiting the LRM's ability to provide accurate guidance.

### Questions
1. It's unclear how the framework robustly handles situations where the OLLM's direct perception strongly contradicts the LRM's reasoning based on the potentially lossy text representation.

2. How significant is the increase in inference time and computational cost when using ThinkOmni compared to running the base OLLM alone?

3. How does the framework handle tasks where the crucial reasoning step relies heavily on non-translatable visual, auditory, or temporal patterns (e.g., recognizing subtle visual defects, interpreting musical harmony, understanding complex physical interactions in a video)? Can specific examples of failure cases due to information loss in text conversion be provided?

4. What happens when the guiding LRM makes a fundamental reasoning error? Does the Stepwise Contrastive Scaling mechanism effectively detect and mitigate the impact of incorrect guidance, or can it lead the OLLM further astray?

5. How sensitive is the performance to the quality of the text summarization/description of the omni-modal input fed to the LRM? Who generates this text, and how is its fidelity ensured?

6. It mentions "maintaining decoding efficiency" (line 104). How is this efficiency achieved when two large models are involved in the decoding loop?

7. How does ThinkOmni perform on tasks requiring common-sense reasoning grounded in multimodal contexts, rather than specialized mathematical or logical reasoning?

### Soundness
3

### Presentation
3

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
This paper proposes the ThinkOmni framework, which aims to enhance the reasoning capabilities of multimodal large language models (OLLMs) by utilizing off-the-shelf large reasoning models (LRMs). By introducing two innovations—LRM-as-a-Guide and Stepwise Contrastive Scaling—ThinkOmni can flexibly balance perceptual and inference signals without additional training, thereby improving multimodal reasoning performance. Experimental results show that ThinkOmni achieves significant performance improvements on multiple benchmark tasks, outperforming existing models that rely on reinforcement fine-tuning.

### Strengths
1. Innovative Framework: The paper proposes ThinkOmni, a training-free, decoding-time method that cleverly integrates a Large Reasoning Model (LRM) into an omni-modal LLM (OLLM) pipeline. The LRM-as-a-Guideand Stepwise Contrastive Scaling mechanisms are novel and well-motivated.
2. Strong Empirical Results: The method shows consistent and significant improvements across six diverse omni-modal reasoning benchmarks. The gains are particularly notable given that no additional training is required.

### Weaknesses
1. Limited Focus on Math Domain:  The benchmark is currently restricted to the math domain at the image-level. It would be beneficial to expand the experiments to include perception-level tasks, such as MME, as well as reasoning benchmarks like MMLU, to provide a more comprehensive evaluation.

2. Confusion Regarding Figure 4: There is some confusion regarding the notation in Figure 4, specifically the terms \( x_{<t} \) and \( x_{<t+1} \). These appear to suggest a multi-turn conversational setup, which warrants further clarification.

3. Minor Performance Gain Relative to ProxyTuning:  The observed performance gain is relatively minor when compared to ProxyTuning, which calls for a deeper analysis of the effectiveness of the proposed approach in this context.

### Questions
Please see weakness.

### Soundness
4

### Presentation
4

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper proposes ThinkOmni, a training-free framework that enhances omni-modal reasoning by leveraging off-the-shelf Large Reasoning Models (LRMs) to guide the decoding process of Omni-modal Large Language Models (OLLMs). The method consists of two main components: LRM-as-a-Guide, which uses textual reasoning models to influence OLLM outputs via logit mixing, and Stepwise Contrastive Scaling, which dynamically balances perception and reasoning signals during inference. The approach is evaluated on six omni-modal benchmarks, showing consistent improvements over baseline OLLMs and competitive performance against reinforcement fine-tuned models.

### Strengths
- The proposed Stepwise Contrastive Scaling mechanism is a meaningful contribution, enabling adaptive tuning of guidance weights without manual hyperparameter search.
- Extensive experiments across six benchmarks demonstrate the generality and scalability of the method.
- The framework is model-agnostic and can be applied to various OLLM and LRM combinations, enhancing its potential impact.
- The idea of leveraging pre-trained LRMs to enhance OLLM reasoning without additional training is novel and practical, especially given the high cost of omni-modal fine-tuning.

### Weaknesses
- The approach requires external LLMs to guide the response. This led to doubts about where the performance gain comes from. Does LLM introduce extra information for solving the question? 
- Although training-free, the approach incurs non-trivial inference overhead due to multiple forward passes per decoding step, which may hinder real-time deployment.

### Questions
- What is the theoretical justification for using Jensen-Shannon divergence to measure disagreement between distributions?  Were other metrics explored?
- How does the method perform when the OLLM and LRM are from different model families or have significant architectural differences?
- How does ThinkOmni handle cases where the LRM provides incorrect reasoning?  Is there a risk of propagating textual reasoning errors into omni-modal outputs?

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
4

### Summary
The paper introduces THINKOMNI, a training-free framework designed to enhance the complex reasoning capabilities of Omni-modal Large Language Models (OLLMs). The authors identify that while OLLMs are good at perceiving diverse modalities (audio, video, image), they often lack the deep reasoning abilities found in recent text-only Large Reasoning Models (LRMs). To bridge this gap without expensive training or data annotation, the authors propose a decoding-time strategy with two main components:
- LRM-as-a-Guide: Leverages off-the-shelf text-only LRMs to guide the OLLM's token generation process via logit fusion.
- Stepwise Contrastive Scaling: A dynamic mechanism that adjusts the guidance weights at each decoding step based on the Jensen-Shannon divergence between different model distributions. This adaptively balances signals from perception (Omni-modal inputs) and reasoning (textual inputs).

Experiments across six multi-modal reasoning benchmarks (including Math Vista, MMAU, and OmniBench) demonstrate that THINKOMNI consistently improves the performance of base OLLMs (like Qwen2.5-Omni), sometimes matching or exceeding models trained with Reinforcement Fine-Tuning (RFT)

### Strengths
- The idea of using a text-only reasoning model to guide an omni-modal model during inference is a clever approach to addressing the scarcity of high-quality omni-modal reasoning data. While guidance decoding exists, applying it specifically to lift reasoning capabilities from one modality to many is a novel application.

- The proposed Stepwise Contrastive Scaling is a technically sound contribution. By using Jensen-Shannon divergence to measure the "disagreement" or unique contribution of reasoning ($D_R$) versus perception ($D_P$) at each step, the model avoids manual hyperparameter tuning and adapts to different task types (e.g., audio perception vs. math reasoning) on the fly.

- The framework is entirely training-free, avoiding the high computational costs and data requirements of Supervised Fine-Tuning (SFT) or RFT. This makes it practically valuable for rapidly deploying improved reasoning in existing omni-modal systems.

### Weaknesses
- (minor, ack'ed by the authors) The framework relies on logit fusion, which strictly requires the OLLM and the LRM to share the exact same tokenizer vocabulary. This is a major limitation to the claim of using "off-the-shelf" LRMs, as it restricts acceptable pairings to models within the same family (e.g., Qwen-based OLLMs with Qwen-based LRMs).

- Since the LRM never sees the omni-modal input and only sees the text trace, there is a theoretical risk that in highly visual/auditory tasks where the text prefix is ambiguous, the LRM might confidently "reason" in a wrong direction based only on text priors. It would be great to study such tricky cases / failure modes.

- While the method beats standard baselines, it would be useful to know if the ~3x compute cost at inference time would be better spent just using a much larger standard OLLM if one is available, rather than guiding a smaller one.

### Questions
See weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

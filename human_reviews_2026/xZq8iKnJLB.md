# EmoFeedback²: Reinforcement of Continuous Emotional Image Generation via LVLM-based Reward and Textual Feedback

- Avg Score: 5.00
- Decision: Reject
- Scores: 4, 6, 4, 6

## Abstract
Continuous emotional image content generation (C-EICG) is emerging rapidly due to its ability to produce images aligned with both user descriptions and continuous emotional values. However, existing approaches lack emotional feedback from generated images, limiting the control of emotional continuity. 
Additionally, their simple alignment between emotions and naively generated texts fails to adaptively adjust emotional prompts according to image content, leading to insufficient emotional fidelity. To address these concerns, we propose a novel generation-understanding-feedback reinforcement paradigm (EmoFeedback²) for C-EICG, which exploits the reasoning capability of the fine-tuned large vision–language model (LVLM) to provide reward and textual feedback for generating high-quality images with continuous emotions. 
Specifically, we introduce an emotion-aware reward feedback strategy, where the LVLM evaluates the emotional values of generated images and computes the reward against target emotions, guiding the reinforcement fine-tuning of the generative model and enhancing the emotional continuity of images. Furthermore, we design a self-promotion textual feedback framework, in which the LVLM iteratively analyzes the emotional content of generated images and adaptively produces refinement suggestions for the next-round prompt, improving the emotional fidelity with fine-grained content. 
Extensive experimental results demonstrate that our approach effectively generates high-quality images with the desired emotions, outperforming existing state-of-the-art methods in our custom dataset. The code and dataset will be released soon.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes EmoFeedback², a novel generation-understanding-feedback reinforcement paradigm for Continuous Emotional Image Content Generation (C-EICG). This method aims to adaptively adjust prompts to control the emotional continuity of the generations. It employs a Large Vision-Language Model (LVLM) to serve as an emotion understanding model. This LVLM provide emotion-aware reward feedback to reinforcement fine-tune the generative model. During inference, it adopts a self-promotion textual feedback framework to iteratively analyze generated images and refine the user prompt during inference. The results on a custom EmoSet-118K based dataset show that the proposed method outperforms other techniques.

### Strengths
### 1. Novel generation-understanding-feedback reinforcement paradigm

The whole paradigm is novel. It trains the LVLM as a reward model for emotion understanding and uses this feedback to guide the image generation. Furthermore, it can adaptively refine the prompt to enhance the emotional control.

### 2. Comprehensive experimental validation
The method achieves superior quantitative performance across five metrics, including V-Error and A-Error,  CLIP-Score and CLIP-IQA, against many modern baselines (EmotiCrafter, EmoEdit, FLUX, SD3.5-L).

### Weaknesses
### 1.The whole pipeline is over complex and heavy
The multi-stage process (diffusion -> RL-train LVLM to produce reward -> GRPO optimization for generation model with LVLM evaluation -> iterative textual feedback with LVLM) is extremely resource-intensive. Even at test time, the iterative self-promotion textual feedback requires multiple LVLM calls per image. Compared with using only the generation model to produce multiple outputs and get the best results, it seems a little unpractical.

### 2. Limited applicability
The model is trained on a custom dataset derived from EmoSet-118K with valence and arousal annotations, which might not generalize well to natural emotional cues. It is also unclear whether the model would perform well on out-of-distribution (OOD) scenarios.

### Questions
1. Sec4.1, which MLLM do you use for generating emotional prompts?
2. Line228 and Line194, you use epsilon for both reward threshold and clipping value of GRPO.
3. What is the batch size during RL training, and the number of epochs?

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
3

### Summary
This paper introduces a new framework for continuous emotional image generation (C-EICG). It tackles the limitations of existing methods, namely their lack of feedback on the generated image's actual emotion and their poor adaptability to user prompts. The core innovation is using a Large Vision-Language Model (LVLM) to provide two types of feedback:
In training, an LVLM evaluates the emotional (Valence-Arousal) values of generated images, providing a reward signal to fine-tune the generative model using reinforcement learning.
In Inference, the LVLM iteratively analyzes images and provides textual suggestions to refine the prompt, improving emotional fidelity.

### Strengths
1. The core generation-understanding-feedback paradigm is a significant contribution. Unlike prior methods that use emotion as a one-way condition, this work creates a closed-loop system. Using an LVLM as both a reward model (for training) and a text-based optimizer (for inference) is a novel and powerful approach.

2. The paper's claims are supported by a set of experiments. Figure 4 and the appendix figures (e.g., 9-11) compellingly illustrate the model's ability to generate smooth and coherent transitions as V-A values change, a key goal of C-EICG that general T2I models (FLUX, SD3.5-L) fail to achieve.

3. The authors effectively demonstrate the necessity of both feedback mechanisms. Figure 5 provides a clear qualitative ablation of RF and TF , and Tables 3 & 4 validate the design choices for the emotion understanding model itself (e.g., model size, reward function, and multi-task training).

### Weaknesses
1. The primary weakness stems from the dataset construction. Both the textual prompts and the crucial V-A labels are synthetic. Prompts are MLLM-generated , and V-A values are sampled from Gaussian distributions derived from discrete emotion categories, not obtained from human annotators. This raises concerns about whether the model is truly aligned with human emotional perception or just with the biases of the synthetic data pipeline.

2. High Inference Cost: The Self-Promotion Textual Feedback framework, while effective, appears to be computationally expensive. The appendix states it uses 3 iterations, generating 8 images per iteration. This iterative process, which requires multiple calls to a 7B LVLM for analysis and prompt optimization , would result in significant latency, making real-time applications difficult.

### Questions
Given that the reward model was trained only on synthetically sampled V-A values, how do the authors expect it to perform on a dataset with genuine human V-A annotations? Is there a risk of domain mismatch, where the model has optimized for the dataset's statistical quirks rather than true emotional representation?

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
This paper proposes a reinforcement-based framework for continuous emotional image generation. The method integrates a Large Vision–Language Model (LVLM) as both a reward model and a textual feedback generator, enabling a closed loop of generation, understanding, and feedback. Through emotion-aware reward optimization and iterative prompt refinement, the approach enhances emotional continuity and fidelity.

### Strengths
1. The generation–understanding–feedback paradigm effectively unites emotional reasoning with reinforcement learning in diffusion models.

2. The textual feedback loop provides an intuitive and interpretable way to refine emotional prompts beyond fixed embeddings.

3. The method achieves the best V-Error, A-Error, CLIP-Score, and user preference rate, validating both emotional fidelity and visual quality.

### Weaknesses
1. The dataset construction strategy closely follows EmotiCrafter (Dang et al., 2025), which already employed an MLLM to generate neutral and emotional prompts as well as Valence–Arousal annotations, resulting in limited methodological novelty.

2. The ground-truth Valence–Arousal annotations are sampled from Gaussian distributions per emotion class, which can introduce label noise and semantic misalignment between images and emotional values.

3. The constructed dataset depends entirely on automatically generated prompts and lexicon-based statistical sampling without any human verification, raising concerns about annotation reliability and perceptual validity.

4. The proposed framework heavily relies on the large vision–language model (LVLM) for both reward computation and textual feedback generation, making the entire pipeline sensitive to model bias and reasoning instability.

5. The self-promotion textual feedback requires iterative LVLM reasoning over multiple generated samples, which substantially increases inference cost, yet the paper does not provide any discussion or analysis of computational efficiency or scalability.

### Questions
1. Given that the ground-truth Valence–Arousal annotations are sampled from Gaussian distributions per emotion class, how do the authors ensure that such synthetic labels accurately reflect the emotional content of the images and do not introduce semantic noise?

2. As the dataset relies entirely on automatically generated prompts and lexicon-based annotations without any human verification, have the authors conducted any manual inspection or validation to assess label reliability and perceptual consistency?

3. The proposed framework depends heavily on the LVLM for both reward computation and textual feedback generation. How robust is the system to the biases or reasoning instability of the LVLM, and could similar performance be achieved with a smaller or less powerful model?

4. The iterative textual feedback mechanism requires multi-image evaluation and textual generation at each step. Could the authors provide quantitative analysis on inference time, computational cost, or scalability to demonstrate its practical feasibility?

5. In Figure 3, the results produced by EmoEdit and EmotiCrafter appear highly similar. Could the authors clarify whether this similarity arises from shared input prompts, overlapping training data, or something else?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes a generation-understanding-feedback reinforcement paradigm for continuous emotional image generation (C-EICG). It leverages a fine-tuned Large Vision-Language Model (LVLM) to address key limitations of existing methods—lack of emotional feedback and insufficient adaptability of emotional prompts—by introducing an emotion-aware reward feedback strategy and a self-promotion textual feedback framework, aiming to enhance emotional continuity and fidelity while maintaining image quality.

### Strengths
1. **Novel Paradigm Design**: The proposed "generation-understanding-feedback" reinforcement framework fills a gap in C-EICG by integrating emotional feedback loops. Unlike existing methods that ignore post-generation emotional evaluation, it uses LVLM’s reasoning ability to close the loop between generation and optimization, bringing a new perspective to emotional image generation.
2. **Well-Integrated Multi-Module Collaboration**: The emotion understanding model (fine-tuned via GRPO), emotion-aware reward feedback, and self-promotion textual feedback are logically coordinated. Each module addresses a specific pain point, and their synergy ensures both emotional accuracy and content consistency, demonstrating a coherent design philosophy.
3. **Practical Training-Free Inference Optimization**: The self-promotion textual feedback framework enables adaptive prompt refinement during inference without retraining the generative model. This design enhances usability, as it can be easily integrated with existing diffusion models (e.g., Stable Diffusion 3.5-Medium) without heavy parameter tuning.
4. **Comprehensive Ablation Studies**: Ablation experiments on LVLM size, reward function design, and single/multi-task training effectively validate the necessity of key design choices. These studies clarify the contribution of each component, strengthening the credibility of the proposed method.
5. **Rich Evaluation Dimensions**: Beyond standard quantitative metrics (emotional accuracy, text-image alignment), the paper includes qualitative comparisons and user studies. This multi-faceted evaluation better reflects the method’s performance in real-world scenarios, aligning with the subjective nature of emotional perception.

### Weaknesses
1. **Limited Discussion on LVLM’s Emotional Evaluation Mechanism**: The paper does not deeply explain how the LVLM (Qwen2.5-VL-7B-Instruct) specifically interprets visual content to assess emotions. The lack of analysis on which visual cues (e.g., color, composition) the LVLM prioritizes makes it hard to understand the mechanistic advantage of using LVLM for emotional feedback.
2. **Insufficient Generalization Validation**: The experiments are primarily conducted on a custom dataset derived from EmoSet-118K. There is no validation on other public C-EICG datasets or cross-domain scenarios (e.g., different image styles, complex scenes), raising questions about the method’s generalizability.
3. **Vague Explanation of Reward Hacking Mitigation**: While the paper mentions using PickScore to avoid content distortion caused by overfitting to emotional cues, it does not detail how PickScore is integrated with the reward function or why it is more effective than other human-preference metrics. This makes the mitigation strategy less transparent.
4. **Lack of Comparison with LVLM-Based Alternatives**: With the rise of LVLM-driven generation optimization, the paper does not compare EmoFeedback2 with other LVLM-aided C-EICG methods (if any) or analyze its unique advantages over general LVLM-based feedback frameworks, weakening the demonstration of its competitiveness.

### Questions
Please refer to the detailed points I raised in the "Weakness" section and respond to each numbered item in your rebuttal with clarifications.

### Soundness
3

### Presentation
3

### Contribution
3

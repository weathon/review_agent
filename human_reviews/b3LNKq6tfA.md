# Learning UI-to-Code Reverse Generator Using Visual Critic Without Rendering

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 6, 5, 5

## Abstract
Automated reverse engineering of HTML/CSS code from UI screenshots is an important yet challenging problem with broad applications in website development and design. In this paper, we propose a novel vision-code transformer (ViCT) composed of a vision encoder processing the screenshots and a language decoder to generate the code. They are initialized by pre-trained models such as ViT/DiT and GPT-2/LLaMA but aligning the two modalities requires end-to-end finetuning, which aims to minimize the visual discrepancy between the code-rendered webpage and the original screenshot. However, the rendering is non-differentiable and causes costly overhead. We address this problem by actor-critic fine-tuning where a visual critic without rendering (ViCR) is developed to predict visual discrepancy given the original and generated code. To train and evaluate our models, we created two synthetic datasets of varying complexity, with over 75,000 unique (code, screenshot) pairs. We evaluate the UI-to-Code performance using a combination of automated metrics such as MSE, BLEU, IoU, and a novel htmlBLEU score. ViCT outperforms a strong baseline model DiT-GPT2, improving IoU from 0.64 to 0.79 and lowering MSE from 12.25 to 9.02. With much lower computational cost, it can achieve comparable performance as when using a larger decoder such as LLaMA.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper proposes ViCT, a multimodal Transformer model to reverse engineer UI code from screenshots. An actor-critic framework is used to train the model, addressing the problem of non-differentiable rendering. The model shows superior results on two novel synthetic UI-to-Code datasets, RUID and RUID-Large. A novel metric, htmlBLUE, is proposed to better compare html code.

### Strengths
1. The paper works on an interesting problem of automated reverse engineering of HTML/CSS code from UI screenshots.
2. The authors propose to formulate the task as a Reinforcement Learning problem to tackle the problem of non-differentiable web rendering.

### Weaknesses
- The paper is not clearly written. Some sections are hard to follow (e.g. Section 3.3). 
- Some parts of the paper are inconsistent. 
  - In Section 4.1, the authors claim to test InstructBLIP[1] as a baseline, but I could not find it in the experimental results.
  - In Section 4.1, the authors mention an experiment "identifying the number of distinct shapes", which is absent in the paper.
- The main and only datasets the authors use for evaluation are fully synthesized. The UIs in the dataset only contain three types of elements, Rectangle, Ellipse and Button. From the examples in Figure 3, I find them quite unrealistic and do not resemble real-world web UIs, which shadows the effectiveness and practical applicability of the model in genuine scenarios.
- Important details on dataset construction and algorithm design are missing (see Questions). 
- Experiments are limited. 
  - Missing baselines, e.g. Pix2Struct [2].
  - The models are only evaluated on two synthetic datasets. Can you run experiments on other datasets, such as the dataset of pix2code [3]?
  - "DiT-LLaMA" is missing in Figure 3.

(Minor)
- In Section 1, 
> In this paper, we take the first step towards reverse-engineering a UI screenshot, i.e., generating an HTML/CSS code that can reproduce the image.

There are prior works on UI-to-Code tasks, such as Pix2Struct[2] and pix2code[3], as you mentioned in Related Works. Do you mean you are the first to directly generate runable UI code without any postprocessing from images?

- Some typos, e.g. a missing period at the end of Section 2.

[1] Wenliang Dai, Junnan Li, Dongxu Li, Anthony Meng Huat Tiong, Junqi Zhao, Weisheng Wang, Boyang Li, Pascale Fung and Steven Hoi. "InstructBLIP: Towards General-purpose Vision-Language Models with Instruction Tuning." arXiv preprint arXiv:2305.06500. 2023.

[2] Kenton Lee, Mandar Joshi, Iulia Raluca Turc, Hexiang Hu, Fangyu Liu, Julian Martin Eisenschlos, Urvashi Khandelwal, Peter Shaw, Ming-Wei Chang and Kristina Toutanova. "Pix2struct: Screenshot parsing as pretraining for visual language understanding." International Conference on Machine Learning. PMLR. 2023.

[3] Tony Beltramelli. "pix2code: Generating code from a graphical user interface screenshot." Proceedings of the ACM SIGCHI Symposium on Engineering Interactive Computing Systems. 2018.

### Questions
1. Please provide more information on the construction of the datasets, RUID and RUID-Large. How do you generate the DOM trees? Which CSS styles do you use as attributes?

2. Please explain the design of the critic model. Is it trained on complete prediction-source pairs and used to estimate values on individual tokens? Additionally, it seems that the critic model only takes visual positional information into account, i.e. IoU. How does the model learn the attributes of the HTML elements, e.g. colors?

3. Please further justify the use of htmlBLEU. In your experiments, you compare htmlBLEU to BLEU with the rendered pixel Mean Squared Error as a standard. Does that mean using MSE as a metric of visual similarity is a better choice?

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 2

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors propose a novel methodology for retrieving HTML/CSS code from screenshots, by stacking a vision encoder, that parses images as sequence of tokens, to a language decoder, that produce the code itself.
To train their netowork, the authors generate a dataset of simple HTML pages with elements and styles, thus enabling large-scale data generation.
To fine-tune the model, the authors rely on a RL algorithm that tries to maximise the similarity between the renders, formalized as a four-class approach. This formulation is differentiable, thus the method can be fine-tuned with gradient descent.
Also, the authors propose the htmlBLEU metric that emphasizes relevant common pieces of HTML/CSS.
Results show that current state of the art creates allucinations, unable to produce similar results to the ground truth.
The authors clarify that this is a proof of concept, and more must be done to get higher-quality results.

### Strengths
1. The paper is original, as it presents an interesrting problem that can be solved through transformers.
2. The state of the art is not able to re-create the same results as the proposal.
3. Interesting technique for generating HTML synthetic data.

### Weaknesses
**Why RL?** I understand that it is not possible, given the render, to propagate gradients to the tokens. However, for the same reason, it is not clear from the paper why this is not a problem when optimising the RL policy. The authors should better explain the passage in 3.3, as now it is very confusing to understand.

**Missing ablation study.** The RL algorithm is given some fixed rewards. How the results changes by varying them? And how these values have been chosen?

**Confusion around the htmlBLEU** While the authors write a generic description of the metric, it would be easier for readers to read an algorithm. Also, the proposed metric does not score too different results with respect to BLEU.

**Synthetic data might be harder to parse than real webpages.** While the introduction of the RUID dataset (and its creation) are very interesting and useful, I argue if the randomness of the approach could generate many samples that are very hard to transform to code, thus impeding the improvement of performance at training time.

### Questions
1. Can the author better explain why they use RL?
2. Can the author provide a better explanation of the htmlBLEU metric?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a framework to process the screenshots of UI and generate related codes based on the LLM decoder. To solve the problem of inefficiency of rendering, a visual critic without rendering (ViCR) module is introduced to predict visual discrepancy of original and generated UI codes. Also, the paper created two synthetic datasets for training and evaluating. An additional metric, named htmlBLEU score, has been developed to evaluate the UI-to-code performance. The proposed method outperforms previous baseline.

### Strengths
1. The paper is well-written and easy to follow.
2. The experimental results are good, demonstrating the effectiveness of proposed method.

### Weaknesses
The method is incremental in terms of scientific research value, just simply modifying the normal pattern of inserting vision encoder into language models. The proposed framework is effective in tackling the UI-to-code generation, but not such a fundamental research in representation learning from my perspective.

### Questions
The paper claims ViCR has no rendering during fine-tuning, but the training objective is based on IoU between reverse-engineered images and the original UI screenshot. So how to acquire the reverse-engineered images without rendering?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper provides ViCT (Vision-Code Transformer), an UI-conditioned code generation model that is fine-tuned with reinforcement learning (RL). More specifically, ViCT takes an UI image as input and generates HTML. ViCT consists of vision foundations models (e.g., ViT and DiT) for encoding images and Large Language Models (LLMs) for generating code. To further align ViCT with the visual similarity between an input UI image and an UI image rendered by generated code, this paper provides ViCR (Visual Critic without Rendering), a reward model for RL fine-tuning. To demonstrate the proof of concept, this paper builds RUID (Random UI Dataset), a new dataset for UI to code generation, that includes about 50K pair of UI image and HTML. With the dataset, this paper shows that ViCT provides comparable performance and fine-tuning with ViCR can further improves the performance.

### Strengths
- S1. The main idea of fine-tuning an image-conditioned text generation model with a reward model and reinforcement learning is very interesting. Even though the concept of an image-conditioned code generation was proposed before, using foundation models (DiT and Llama) and fine-tuning the model with RL (Policy Gradient method) seems novel.

- S2. To demonstrate the proof of concept, this paper builds a new dataset for UI to code generation, which contains about 50K pairs of UI and HTML (RUID-Large, Random UI Dataset).

### Weaknesses
- W1. Overall architecture of the proposed method (ViCT) seems reasonable. However, I am not sure that the design choice for the reward modeling and RL fine-tuning is effective. The overall method is similar to Reinforcement Learning with Human Feedback (RLHF), a recent prevailing method for LLM alignment. In RLHF, the reward model (RM) is usually modeled by relative feedback (preference or superiority) over a pair of inputs. Also, the prevalent RL algorithm is Proximal Policy Optimization (PPO) rather than vanilla Policy Gradient (PG). It would be better to provide some considerations on these design choices. And, it would be much better to provide a comparison between ViCR (absolute feedback + PG) and RLHF methods (relative feedback + PPO).

- W2. I am not sure how effectively ViCR models an intermediate reward in Eq 2. According to Eq 2., \hat{q_theta}(w_t^s), a value function for the token w_t^s is used. Can the reward model (ViCR) estimate the value for an intermediate token in partially generated code?

### Questions
- Q1. Regarding W1, how does the reward model (ViCR) perform? Since this paper models ViCR as classification of visual similarity (very low, low, high and very high), it will be better to provide classification accuracy.

- Q2. Regarding W1, how does the learning curve (e.g., IoU over learning steps) look like? It will help readers to understand the learning dynamics in RL fine-tuning of ViCT.

- Q3. Regarding W2, how does the token-level reward model perform?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

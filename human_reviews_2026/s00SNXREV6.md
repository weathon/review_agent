# Vision-Zero: Scalable VLM Self-Evolution via Multi-Agent Self-Play

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 4, 8

## Abstract
Although reinforcement learning (RL) can effectively enhance the reasoning capabilities of vision–language models (VLMs), current methods remain heavily dependent on labor-intensive datasets that require extensive manual construction and verification, leading to extremely high training costs and consequently constraining the practical deployment of VLMs. 
To address this challenge, we propose **Vision-Zero**, *a domain-agnostic self-play framework that generates visual deduction games from diverse images for scalable VLM training without human annotations.*
Specifically, Vision-Zero encompasses three main attributes:
(1) **Strategic Self-Play Framework:**
Vision-Zero trains VLMs in "Who Is the Spy"-style games, where the models engage in strategic reasoning and actions across multiple roles. Through interactive gameplay, models autonomously generate their training data without human annotation.
(2) **Gameplay from Arbitrary Images:** Unlike existing gamified frameworks, Vision-Zero can generate games from arbitrary images, thereby enhancing the model’s reasoning ability across diverse domains and showing strong generalization to different tasks.
We demonstrate this versatility using three distinct types of image datasets: CLEVR-based synthetic scenes, charts, and real-world images.
(3) **Sustainable Performance Gain:** We introduce Iterative Self-Play Policy Optimization (Iterative-SPO), a novel training algorithm that alternates between Self-Play and reinforcement learning with verifiable rewards (RLVR), mitigating the performance plateau often seen in self-play-only training and achieving sustained long-term improvements.
Despite using label-free data, Vision-Zero achieves state-of-the-art performance on reasoning, chart question answering, and vision-centric understanding tasks, surpassing other annotation-based methods.
Models and code will be released upon acceptance.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes Vision-Zero, a zero-human-in-the-loop post-training framework for VLMs built upon a visual “Who is the Spy?” self-play game. This paper further proposes iterative self-play optimization for this self-play game task, which consists of self-play optimization in clue stage and RLVR for the decision stage. Experiment demonstrate consistent gains over baseline methods for Qwen2.5-VL-7B base model.

### Strengths
1.	This paper demonstrates that self-play visual game can be an effective task for VLM post-training, which is an interesting finding.

2.	The paper evaluates on a reasonably wide set of benchmarks to demonstrate the effectiveness of Vision-Zero.

### Weaknesses
1.	Prior work has already explored automatically crafted multi-image contrastive data for VLM post-training (e.g., MiCo). The paper should avoid claiming to be the first zero-human-in-the-loop paradigm and more carefully situate its contribution relative to [1].

2.	The “Who is the Spy?” pipeline requires multiple forward passes per sample and a more complex training algorithm. To justify this complexity, the paper should include direct, apples-to-apples comparisons against simpler multi-image contrast baselines such as MiCo [1], using the same data sources (and ideally matched compute) to test whether the game mechanics yield additional gains beyond contrastive RLVR.

3.	Please report training efficiency for Vision-Zero. The multi-turn interactions per sample could introduce substantial overhead; quantifying this is important for practical adoption.

4.	The main text states 100 iterations while the appendix reports 150. Please reconcile this discrepancy and clarify which results correspond to which training steps.

[1] MiCo: Multi-image Contrast for Reinforcement Visual Reasoning. NeurIPS 2025

### Questions
Please see Weaknesses

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
4

### Summary
This paper introduces Vision-Zero, a framework inspired by the “Spy vs. Player” game concept. The idea is creative and brings an interesting self-play strategy from LLMs into the vision-language domain (VLMs). The design is simple yet somewhat innovative. However, the overall performance improvement is modest and appears constrained by the capability of the underlying image editing models.

### Strengths
1.	The paper identifies two valuable and timely challenges in multimodal learning: data scarcity and knowledge ceiling.
2.	It presents a clear motivation by extending the concept of self-play from large language models (LLMs) to vision-language models (VLMs).
3.	The framework generates data using automated image editing tools or procedural rendering, which supports scalable dataset creation.
4.	The Spy–Player game formulation is an interesting and original idea that provides a novel self-supervised interaction mechanism.

### Weaknesses
1.	The paper identifies two valuable and timely challenges in multimodal learning: data scarcity and knowledge ceiling.
2.	It presents a clear motivation by extending the concept of self-play from large language models (LLMs) to vision-language models (VLMs).
3.	The framework generates data using automated image editing tools or procedural rendering, which supports scalable dataset creation.
4.	The Spy–Player game formulation is an interesting and original idea that provides a novel self-supervised interaction mechanism.

### Questions
1.	How fair is the comparison with prior works that do not assume access to camera poses? How does MultiViewPano perform when camera poses are inaccurate or unavailable?
2.	How robust is the proposed method to poor-quality SEVA outputs? Can the authors provide visualizations of such cases?
3.	Could the authors expand the ablation study and visualize intermediate outputs to clarify which modules contribute to which improvements?
4.	How are geometric distortions and artifacts handled during repeated enhancement steps?
5.	Why does the proposed method not outperform CubeDiff, especially for indoor scenes?
6.	Could the authors ensure that all best-performing metrics in Table 1 are boldfaced for readability?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces a zero-human-in-the-loop training framework for VLMs using a gamified self-play mechanism inspired by social deduction games like "Who Is the Spy?", eliminating the need for expensive human-annotated datasets.
The game involves multiple agents (civilians and a spy) who observe slightly different images and must describe or deduce the differences through dialogue. The model improves via Iterative Self-Play Policy Optimization (Iterative-SPO), which alternates between self-play and reinforcement learning with verifiable rewards (RLVR) to avoid performance plateaus.

### Strengths
1. First to apply self-play to VLMs in a gamified adversarial setting.
2. Propose a novel training algorithm that avoids equilibrium stagnation by alternating between self-play and RLVR.
3. The evaluated domains are quite comprehensive, including math, charts, vision-centric, and simulated images.

### Weaknesses
- The improvement of the Chart/OCR is marginal, as shown in Tab.2.
- The data preparation cost in the Tab.3 is ambiguous, where the statistics might be further explained.
- The improvement over the InternVL series is marginal compared to the QwenVL series, as shown in Tab. 4, casting doubts on its generability.
- The intuition/theory between the "who is a spy" proxy task and the general question-answering tasks remains unclear.

### Questions
- Will the models cheat or find shortcuts via editing methods? Because the editing possibilities/categories are pre-defined. (e.g., swapping numerical attributes in chart data.)
- Is this idea novel in the multi-agent community? I have noticed that Role-Advantage Estimation is based on the previous research. What's your differences?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The paper presents VISION-ZERO, a framework that enhances the reasoning capabilities of vision–language models (VLMs) through self-play and RLVR, thereby reducing reliance on costly human-annotated data. Specifically, the framework constructs a strategic environment inspired by the game “Who Is the Spy”, in which training data are naturally generated through self-play. The resulting data are then used to iteratively optimize VLMs using the Iterative-SPO algorithm. Extensive empirical results on multiple benchmarks demonstrate that VISION-ZERO surpasses baseline models, significantly enhancing the reasoning ability of VLMs while avoiding cross-capability negative transfer and substantially reducing dataset construction costs.

### Strengths
1. This paper is well-motivated. Existing RLVR methods rely heavily on carefully constructed datasets. By introducing self-play, this method enables reasoning improvement on label-free data, helping to extend RLVR to broader domains where manual annotation is difficult or impractical.
2. The experiments are comprehensive. Evaluations on **Task Generalization Capability**, **Cross-Capability Negative Transfer Mitigation**, and **Low Dataset Construction Costs** strongly demonstrate the superiority of **VISION-ZERO**, providing solid evidence for its effectiveness in mitigating data scarcity and improving scalability.
3. The framework proposed in this paper is novel. The *“Who Is the Spy”* game environment appears to be highly scalable and can be readily extended to diverse domains. This offers a new paradigm for enhancing the capabilities of vision–language models (VLMs).

### Weaknesses
1. The data curation pipeline is primarily focused on the **CLEVR** and **Chart** domains. As mentioned in *Appendix 2.2*, directly editing chart images using **NanoBanna** and **ChatGPT** can be extremely challenging. This raises concerns about the scalability of the proposed framework, particularly regarding how to generate image pairs with arbitrary data.
2. The overall design of this framework appears to be sophisticated. The success of the framework relies on the effective coordination among **self-play in the game environment**, **Self-Play Optimization in the Clue Stage**, and **RLVR in the Decision Stage**. A deeper discussion on the stability of this framework would be valuable for understanding its robustness and generalization.

### Questions
1. Ablation studies on critical components and hyperparameters (e.g., RAE and the number of civilians) would be helpful for further understanding the contribution of each module and the sensitivity of the framework.
2. If **VISION-ZERO** is trained on static datasets, is there a possibility of **knowledge leakage** when the model encounters the same image pairs during training?

### Soundness
3

### Presentation
4

### Contribution
3

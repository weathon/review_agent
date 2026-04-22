# Dynamic Classifier-Free Diffusion Guidance via Online Feedback

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 2

## Abstract
Classifier-free guidance (CFG) is a cornerstone of text-to-image diffusion models, yet its effectiveness is limited by the use of static guidance scales. This ``one-size-fits-all'' approach fails to adapt to the diverse requirements of different prompts; moreover, prior solutions like gradient-based correction or fixed heuristic schedules introduce additional complexities and fail to generalize. In this work, we challenge this static paradigm by introducing a framework for dynamic CFG scheduling. Our method leverages online feedback from a suite of general-purpose and specialized small-scale latent-space evaluators—such as CLIP for alignment, a discriminator for fidelity and a human preference reward model—to assess generation quality at each step of the reverse diffusion process. Based on this feedback, we perform a greedy search to select the optimal CFG scale for each timestep, creating a unique guidance schedule tailored to every prompt and sample. We demonstrate the effectiveness of our approach on both small-scale models and the state-of-the-art Imagen 3, showing significant improvements in text alignment, visual quality, text rendering and numerical reasoning. Notably, when compared against the default Imagen 3 baseline, our method achieves up to 53.8% human preference win-rate for overall preference, a figure that increases up to to 55.5% on prompts targeting specific capabilities like text rendering. Our work establishes that the optimal guidance schedule is inherently dynamic and prompt-dependent, and provides an efficient and generalizable framework to achieve it.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper introduces a new method to **dynamically regulate** the classifier-free guidance (**CFG**) scale in text-to-image diffusion models. At each denoising time step a guidance scale is greedily selected among a predefined set based on the feedback scores coming from different **evaluators**. These evaluators take in input a time step $t$ and a latent $x_t$ and evaluate: (a) the alignment to the text prompt, (b) the visual quality of the output, (c) the alignment w.r.t. human preferences, and optionally the quality of (d) text rendering or (e) numerical reasoning. The method is evaluated by experimenting different base models (LDM and Imagen 3) and different text prompt datasets (Gecko, GenAI-Bench, MARIO, GeckoNum). Both quantitative and qualitative results overall prove the effectiveness of the method and of each evaluator.

### Strengths
1. **Interesting use of the CFG scale.** I find the idea of using CFG scale to improve the alignment and the quality of text-to-image generative models very interesting and not too explored in the research community. I think the paper would definitely provide value to the ICLR community and inspire similar research directions. 
2. **Interesting experiments.** In addition to the main contribution, I find the experiments reported in Figure 2 very interesting as they show how different CFG scales influences different metrics at different denoising time steps.
3. **Clear and well-structured presentation.** The paper is well structured and easy to follow.

### Weaknesses
1. **Ambiguity in algorithm description.** Lines 216–220 could be misread as computing all CFG outputs in parallel and assembling them afterward. Clarifying that the selection at each step influences subsequent sampling would be helpful.
2. **Partial evaluation of latent evaluators.** Section 5.1 reports detailed analysis only for the CLIP evaluator; others (e.g. discriminator) are not explicitly benchmarked.
3. **Model-agnostic claim.** I do not necessarily agree with the claim that the method is *model-agnostic* (line 428). Similarly to how baselines which rely on heuristics have to run hyper-parameter search to find the optimal CFG scale for a specific model, the proposed method also has to train the evaluators on the noisy latents coming from each specific model.

### Questions
Following the weakness listed above, I would like to ask these questions:
1. Lines 216–220 could be misread as computing all CFG outputs in parallel and assembling them afterward. Could this part be explained better?
2. Section 5.1 is named *Evaluation of latent evaluators*, but in practice only evaluates the performances of the latent CLIP evaluator and not of the others. Would it be possible to provide a justification of this choice?
3. Some minor issues:
    - Equation 6 refers to $W$ which, if I am not wrong, is never defined before. May this be $S$?
    - In Equation 7 the alpha coefficients could potentially assume negative values. Should there be an absolute value around the difference between $e_{t}$ and $e_{t+1}$?

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
2

### Summary
* The paper introduces a framework for dynamic CFG in T2I diffusion models.
Unlike conventional static CFG scales or fixed heuristic schedules, the proposed method adaptively adjusts the CFG scale at each sampling step using online feedback from multiple evaluators.
* These evaluators include a CLIP-based alignment model, a discriminator for visual fidelity, a human preference reward model, and specialized evaluators for text rendering and numerical reasoning.
* The framework employs a greedy search to select the optimal CFG scale per timestep, generating a prompt-dependent schedule.
* Experiments on both SD models (LDM) and the Imagen 3 show consistent improvements across alignment, visual quality, text rendering, and counting tasks.

### Strengths
* The idea of making CFG dynamic and feedback-driven is interesting.
It challenges the assumption that CFG scales should remain static during generation.
* Operating in the latent space keeps the method lightweight, making it feasible for real-world use.
* The framework transfers successfully from diffusion models to large-scale proprietary ones like Imagen 3, showing robustness and adaptability.
* The authors assess their method across diverse benchmarks combining quantitative metrics and human preference studies.
* Visualizations of dynamic schedules clearly show how guidance strength varies with prompt complexity, enhancing transparency and interpretability of the approach.

### Weaknesses
* I believe per-step greedy search does not guarantee global optimality.
The method assumes independence between timesteps, which may overlook temporal correlations in the diffusion trajectory.
* The comparisons may be a bit limited.
Including stronger recent baselines, like RL-based or multi-step optimization methods (FK steering or diffusion RL), would make the results more convincing.
* In my understanding, each evaluator must be re-trained for the specific diffusion model, which weakens the claim of being fully model-agnostic.

### Questions
* How does the method ensure temporal consistency of guidance across timesteps given the greedy selection strategy?
* Could a differentiable or policy-gradient optimization approach yield smoother and potentially superior CFG schedules?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper proposes an approach, Dynamic CFG, for dynamically choosing the classifier-guidance scale at each individual timestep using a series of online evaluators. By choosing the optimal cfg-scale with respect to the alignment/visual quality/human preference score, Dynamic CFG improved overall generation quality with adaptive evaluators’  weight method.

### Strengths
**Clear motivation and well-written:** The proposed method is clear motivated and the writing of this paper is great. Most of terms have been clearly explained. It is conformable to this paper. 

**Lightweight evaluator design:** The latent evaluator is lightweight and can be seamlessly integrated into existing diffusion models without significantly increasing computational cost.

**Strong effectiveness:** The method demonstrates consistent improvements across multiple benchmarks, and it could enhance text alignment, visual quality, and human preference win rates on strong baselines like LDM and Imagen 3.

### Weaknesses
**High training cost for evaluators:** The method requires training multiple specialized latent evaluators with dedicated training datasets. This increases the overall implementation cost and limits the practicality of applying the approach to new tasks (e.g., when user wants to enhance the layout alignment).

**Dependence on evaluator quality:** The performance of the proposed method heavily relies on the accuracy and generalization ability of the evaluators. One poorly trained evaluator may lead to unstable or suboptimal guidance schedules. The authors could add more discussion for this question to increase the transparency of the proposed method.

### Questions
1. While the online evaluator is lightweight and the inference cost is low, and the current choice of five CFG-scale candidates proves effective, it remains unclear how much additional performance gain could be achieved by introducing more candidate scales.

2. Transferability: For the latent evaluator, can it be transferred to other models trained in the same latent space?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces a dynamic Classifier-Free Guidance (CFG) scheduling framework for text-to-image diffusion models, replacing the traditional static, one-size-fits-all guidance scale with adaptive, prompt-specific schedules. The method uses latent-space evaluators (e.g., CLIP, discriminator, and human preference reward models) to provide real-time feedback on alignment, fidelity, and reasoning quality during the diffusion process. A greedy search dynamically selects the optimal CFG scale at each timestep, yielding schedules that adapt to both the prompt and diffusion stage. Experiments on StableDiffusion-like models and Imagen 3 show substantial gains in text alignment, visual quality, and human preference, proving that optimal CFG is inherently dynamic and prompt-dependent.

### Strengths
The authors identify a interesting research question: how to improve generated images across multiple aspects such as alignment, fidelity, et, by dynamically adapting the Classifier-Free Guidance (CFG) scale instead of relying on a static, fixed value.

### Weaknesses
1. The approach heavily depends on CLIP image encoder fine-tuning under noisy latent conditions, which may distort the original embedding space; however, the paper provides neither validation nor ablation studies to justify or quantify this effect.

2. The experimental comparison is limited to conventional baselines. For example, [1] introduces a test-time optimization strategy designed for multi-reward settings. The authors may need to compare their method with this. In additions, considering that the method employs multiple specialized evaluators (e.g., human preference, counting) and substantial fine-tuning, it can also be compared against multi-objective RL-based diffusion approaches, which incur similar computational costs.

3. The assumption that a greedy-selection-based dynamic CFG universally enhances performance across different aspects lacks both theoretical grounding and intuitive justification; additional analysis or formal reasoning is necessary.

4. The overall writing quality could be improved for clarity, coherence, and readability.

[1] Test-time Alignment of Diffusion Models without Reward Over-optimization, Kim et al., ICLR 2025

### Questions
It is unclear what $W$ represents in Equation (6); the authors should explicitly define it.

The relationship between the guidance scale $s$ and the evaluator score $e_t$  needs clarification. Is $e_t$ defined as an aggregate across multiple evaluators?
If $e_t$ is indeed defined as the average over different evaluator scores, this should be explicitly stated, along with the precise formulation.

The notation $E$ in Equation (7) is also ambiguous. The authors should clarify this as well.

### Soundness
1

### Presentation
2

### Contribution
1

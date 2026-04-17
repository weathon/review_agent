# ViMo: A Generative Visual GUI World Model for App Agents

- Decision: Accept (Poster)
- Scores: 8, 6, 4, 6

## Abstract
App agents, which autonomously operate mobile Apps through GUIs, have gained significant interest in real-world applications. Yet, they often struggle with long-horizon planning, failing to find the optimal actions for complex tasks with longer steps. To address this, world models are used to predict the next GUI observation based on user actions, enabling more effective agent planning. However, existing world models primarily focus on generating only textual descriptions, lacking essential visual details. To fill this gap, we propose ViMo, the first Visual world Model designed to generate future App observations as images. For the challenge of generating text in image patches, where even minor pixel errors can distort readability, we decompose GUI generation into graphic and text content generation. We propose a novel data representation, the Symbolic Text Representation (STR), to overlay text content with symbolic placeholders while preserving graphics. With this design, ViMo employs a STR Predictor to predict future GUIs’ graphics and a GUI-text Predictor for generating the corresponding text. Moreover, we deploy ViMo to enhance agent-focused tasks by predicting the outcome of actions. Experiments show that ViMo establishes visual world models as a compelling alternative to language-based approaches, producing visually plausible and functionally effective GUIs that empower App agents with more informed decisions.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents ViMo, a generative visual world model for mobile GUI agents. It addresses the failure of text-only world models to capture essential visual details and the failure of general image models to render text legibly. The paper's core idea is to decouple GUI generation: a diffusion-based *STR Predictor* generates the non-text graphical layout, and an LLM-based *GUI-text Predictor* generates the text content for the predicted layout. ViMo is then used as a lookahead mechanism to help a base agent select the best action from a list of candidates, leading to improved decision-making accuracy.

### Strengths
- A Novel and Necessary Model. The idea of a visual world model for GUIs is a logical and necessary next step for the field. The authors are one of the first to propose a concrete and workable architecture for this.
- The proposed STR is a good workaround to decompose the problem, allowing the diffusion model and the LLM to handle the sub-tasks they are respectively good at (spatial graphics vs. semantic text).
- Clear Downstream Task Improvement. The paper doesn't just evaluate image quality; it demonstrates a clear, practical benefit. Using ViMo to "preview" actions improves the underlying agent's step accuracy by a significant margin. The online task success rate also improves by an absolute 7.76%

### Weaknesses
- Complexity in architecture. The framework is a complex multi-stage pipeline (OCR -> STR creation -> Diffusion prediction -> Symbol detection -> LLM text prediction -> Rendering). A failure at any one of these steps (e.g., the initial OCR misses text, the diffusion model generates a malformed placeholder, the symbol detector fails) could cause the entire prediction to fail. Can the authors elaborate more on how to effectively build an end-to-end world model in the future? What are the fundamental challenges, and is it just data? It'll also be nice to see any initial effort or blueprint for that kind of world models.
- With 20K images, ViMo is already with pretty good capability. Do the authors have any data-centric scaling experiments or ideas if we can have more samples by simply scaling the environment and explore? Also, what would be the challenges to train a more universal digital world model that can also predict on high-res desktop envs?

### Questions
Please see weaknesses.

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
**Summary**

This paper introduces **ViMo** (Visual GUI World Model), a novel **generative world model** designed for **App agents** that interact with mobile graphical user interfaces (GUIs). Current App agents often struggle with **long-horizon planning**—they cannot accurately predict how future screens will look after performing an action. Prior “world models” attempt to simulate such outcomes but typically rely on **text-only descriptions**, which fail to capture visual layouts and pixel-level details essential for GUI reasoning.

To overcome these limitations, ViMo formulates the problem as **visual GUI prediction**: given a current screen (x_k) and a user action (a), it generates the **next GUI image** (x_{k+1}^a = f(x_k, a)). The model’s core innovation is the **Symbolic Text Representation (STR)**, a hybrid representation that decouples **graphic** and **textual** content generation. STR replaces all text on a GUI with rectangle-shaped placeholders (“text symbols”), allowing ViMo to first predict the overall layout and visual structure (using a **diffusion-based STR predictor**) and then fill in textual content (via a **LLM-based GUI-text predictor**). This design mitigates pixel-level rendering errors that commonly distort text in diffusion models.

ViMo is evaluated in three major settings:

1. **World-model ability:** ViMo achieves superior GUI generation quality compared with baselines such as IP2P*, HTML-vision, and UI-diffuser, measured by GUI consistency ((s_{gc})), instructional accuracy ((s_{ia})), and action readiness ((s_{ar})) under both automatic metrics and user studies.
2. **Agent-enhanced decision making:** When integrated into existing App agents (e.g., T3A, M3A), ViMo improves step-wise action prediction accuracy by up to **14.07%** and raises online task completion rates by **7.76%**.
3. **Real-world deployment and generalization:** ViMo runs efficiently (≤16 GB GPU, plug-and-play API) and generalizes to unseen Apps in zero-shot evaluations, achieving 47.6% step accuracy.

**Contributions:**

* Proposes **ViMo**, the **first generative visual GUI world model** capable of predicting future App observations as realistic images.
* Introduces **Symbolic Text Representation (STR)** to separate layout prediction from text generation, ensuring high visual fidelity and readable on-screen text.
* Demonstrates through extensive experiments that ViMo significantly enhances both **world-model quality** and **App-agent decision-making performance**, outperforming prior language- and vision-based baselines across multiple benchmarks.

### Strengths
The paper proposes a **novel and well-motivated idea** that effectively addresses the weakness of existing generative models in handling text within GUI scenes. The introduction of **Symbolic Text Representation (STR)** is both elegant and practical, enabling clear separation between visual layout generation and text rendering.

Experiments are **comprehensive and convincing**, demonstrating strong improvements in both GUI generation quality and downstream App-agent performance. The results show that ViMo not only produces more realistic and legible interfaces but also enhances action prediction and task success rates.

Overall, the work is **original**, **technically solid**, and **highly significant**, offering a new perspective on world modeling for digital interfaces with clear implications for multimodal agents.

### Weaknesses
While the paper’s idea is strong, the **experimental setup lags behind the current state of the field**. Most evaluations rely on relatively **outdated models and benchmarks**, such as older App-agent frameworks and base models that no longer reflect the best-performing systems. Recent models (e.g., **UITars**, **Qwen3-VL**, and other 2025-era multimodal agents) have reported around **60% step accuracy on AndroidWorld**, significantly higher than the baselines used in this paper. As a result, it is difficult to fully gauge how ViMo would perform when integrated with these newer, stronger agents.

### Questions
* Could the authors update the experiments by integrating **ViMo** with more recent models such as **UITars** or **Qwen3-VL**, which achieve around 60% accuracy on AndroidWorld?

* With the rapid progress of modern **image and multimodal generation models** (e.g., *Qwen-Image-Edit, Nano-Banana, GPT-Image*), the gap in text rendering and visual fidelity is quickly shrinking. Do the authors envision a future where **digital world models** can be trained **end-to-end**, without the need for modular designs like STR and text-filling stages?

* Regarding the **conceptual validity of digital world models**—humans can intuitively predict outcomes in the *physical* world, but GUI environments are often highly dynamic, with frequent content refreshes and animations that even users cannot anticipate precisely. How do the authors view the **epistemic limits** of modeling such digital environments? Do the authors see ViMo’s predictive modeling as primarily a *functional approximation* for agents, rather than a realistic simulation of human prediction?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper proposes ViMo, the first visual GUI world model that predicts future app observations as images rather than text descriptions. The key innovation is the Symbolic Text Representation (STR), which overlays text content with rectangular placeholders to decompose GUI generation into graphics and text generation. ViMo uses a diffusion-based STR Predictor for graphics and an LLM-based GUI-text Predictor for text content. Experiments demonstrate improvements in GUI quality metrics and app agent decision-making performance.

### Strengths
**(1) First visual world model formulation for GUI agents:**
ViMo is among the first works to explicitly formulate GUI state prediction as a generative visual world model, extending diffusion-based modeling beyond robotics and embodied domains to mobile GUI environments.

**(2) Novel STR Representation:**
The proposed Symbolic Text Representation (STR) elegantly addresses the challenge of text generation in GUIs by decoupling location prediction (via diffusion) from content generation (via LLM).
This decomposition is technically sound, mitigates the multimodal alignment issue highlighted in Fig. 1, and may generalize beyond GUI generation to other structured visual reasoning tasks.

**(3) Integration with existing App agents and measurable improvements:**
ViMo is integrated as a plug-in to current App agents (T3A, M3A) and achieves consistent step-level accuracy gains (+9–14%) as well as moderate task-level improvement (+7.8%) on AndroidWorld.
These results demonstrate that visual world models can effectively complement decision-making modules within agent pipelines.

**(4) Thorough experimental analysis:**
The paper provides detailed evaluations, including trajectory synthesis, error-accumulation analysis, qualitative failure cases, and inference-time comparison.
This level of transparency and comprehensive reporting is commendable and contributes to the reproducibility of the study.

### Weaknesses
**(1) Limited evaluation data scale:**
The main experiments rely on relatively small-scale or sampled settings, including 19 apps with 57 episodes for world-model evaluation and 116 tasks for online testing.
This limited scope raises concerns about statistical significance and makes it difficult to assess robustness across diverse GUI domains and app categories.

**(2) Evaluation method validity concerns:**
Most reported metrics depend on LLM-as-judge evaluations and a small-scale human study (~20 samples per method).
While these methods provide qualitative insights, they introduce potential bias, subjectivity, and circularity, especially since the same class of LLMs is used for both generation and evaluation.

**(3) Modest performance gains with high computational cost:**
As shown in table 12, ViMo achieves 49.20% step accuracy, only 2.56 points higher than the Change-Text baseline, while requiring approximately two minutes per inference, about 24 times slower than the five-second language baseline.
Such computational overhead raises concerns about the model’s practical applicability given the modest improvement in accuracy.

**(4) Limited competitiveness against existing GUI agents in other paradigms:**
Compared with the current AndroidWorld leaderboard (https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?gid=0#gid=0), where most end-to-end or grounding-based agents complete steps in under 10 seconds and achieve up to 97.4% accuracy, ViMo’s 40.95% task accuracy and significantly higher latency make it less competitive.

**(5) Presentation and clarity issues:**
- The paper’s organization is dense and at times difficult to follow. 
- Some figures (e.g., Fig. 1 and Fig. 2) are overly compact, and the dense layout and small text reduce overall readability and visual clarity.
- The spacing of several tables (e.g., between Tables 2 and 4–5, and below Table 7) is overly tight, which reduces readability and makes the layout appear visually crowded.
- The appendix is not properly referenced or linked from the main text, making it difficult for readers to locate the corresponding details without manual navigation.

### Questions
**(1) Comparison with GUI Agent in other paradigm:**
Given that most end-to-end or grounding-based agents on the current AndroidWorld leaderboard achieve up to 97.4% accuracy with per-step latency below 10 seconds, while ViMo reaches 40.95%, could the authors elaborate on the intended advantages or use cases of adopting a world-model approach in this context?
In particular, under what conditions might such a model complement or outperform grounding-based or end-to-end systems?

**(2) Cost–benefit analysis:**
ViMo achieves only +2.56% improvement over Change-Text while being roughly 24× slower (2 min vs 5 s). For a 10-step task, this translates to about 20 minutes vs 50 seconds.
What use cases justify this trade-off? Have you explored optimizations or hybrid approaches to reduce inference latency?

**(3) Dataset and evaluation scale:**
If only the subset of 57 episodes was used, what criteria were applied when sampling these episodes to ensure representativeness across app types?
Could the authors expand the evaluation to include a larger dataset to strengthen statistical validity and demonstrate robustness across different GUIs?

**(4) Evaluation methodology:**
Since most metrics rely on LLM-as-judge evaluations, have the authors examined inter-rater consistency or any correlation between LLM-based scores and human evaluations?
Such analysis would help validate whether the automatic metrics reliably reflect real-world agent performance.

**(5) Presentation:**
Please consider addressing the issues outlined in the weaknesses section, such as improving figure clarity, table spacing, and better cross-referencing between the main text and appendix.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces a generative visual world model for mobile app agents that predicts future GUI observations as images rather than textual descriptions. The core idea is to improve long-horizon planning for agents that control mobile apps, by visually simulating the outcomes of user actions. Because rendering text is challenging for diffusion models, the authors decompose generation into graphics (via diffusion) and text (using an LLM). Experiments show that ther proposed method improves the realism of generated GUI screens, particularly the text elements. Using the method for planning improves agent's accuracy (+14%), and generalizes to unseen apps. The overall contribution is a visual diffusion+LLM GUI simulator as an alternative to language-based GUI simulation.

### Strengths
- introduces a visual world model that predicts future app GUI observations as images. This is unlike prior language-based world models that generate only text descriptions of GUIs.
- proposes an “STR” representation which overlays all textual content in the GUI with symbolic placeholders (white rectangles), This simplifies the text generation task into text localization while the graphics are generated with a diffusion model
- Proposes two decoupled modules: 1) one generates future GUI structure as an image with placeholders for text (white rectangles) 2) the other one (LLM-based) fills in text content for each placeholder rectangle.
- Good experimental results:
   - Better GUI world model: achieves a significant improvement in automatic GUI quality metrics and in user studies versus existing world models 
   - also boosts action prediction accuracy, and, to a lesser degree, online task completion rate

### Weaknesses
The paper is generally well written, but there are some sections that are not clear or lack important details
- The dataset section is quite short and does not explain the task that the agent performs
- Why is the dataset subsampled from the larger one? Why not use a standard dataset split?
- Is the 'dataset summarization' referring to training data or test data
- what is "split summarisation"?
- Unclear how successful actions are determined
- Details on how user actions are encoded for the diffusion process are vague

(Minor)
- Table captions do not specify what the columns mean, eg accuracy? 
- Typo in "Table 3: Comapre World Models"

### Questions
1) Why is the dataset subsampled from the larger one? 
This begs the question of whether the test set was selected in such a way as to favor the proposed method over others. Why not use a standard dataset split?

2) Please explain what is the task that the agent must perform
Does the input start with an empty screen? Is the instruction given, e.g. "set an alarm for 9am"? 

3) (Most important question) Will this type of approach be practical?

The paper says "Inference time on V100 GPU is 8 seconds on a STR image generation and 30 seconds on GUI-text prediction." However the application is an agent that operates mobile apps. Mobile devices don't have GPUs! How can this method be practically applied? The "online evaluation" implies that the method was applied in real time, but there are very few details.

### Soundness
3

### Presentation
2

### Contribution
3

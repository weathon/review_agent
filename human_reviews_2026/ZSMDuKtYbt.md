# SketchThinker-R1: Towards Efficient Sketch-Style Reasoning in Large Multimodal Models

- Decision: Accept (Poster)
- Scores: 6, 6, 6

## Abstract
Despite the empirical success of extensive, step-by-step reasoning in large multimodal models, long reasoning processes inevitably incur substantial computational overhead, i.e., in terms of higher token costs and increased response time, which undermines inference efficiency. In contrast, humans often employ sketch-style reasoning: a concise, goal-directed cognitive process that prioritizes salient
information and enables efficient problem-solving. Inspired by this cognitive efficiency, we propose SketchThinker-R1, which incentivizes sketch-style reasoning ability in large multimodal models. Our method consists of three primary stages. In the Sketch-Mode Cold Start stage, we convert standard long reasoning process into sketch-style reasoning and finetune base multimodal model, instilling initial sketch-style reasoning capability. Next, we train SketchJudge Reward Model, which explicitly evaluates thinking process of model and assigns higher scores to sketch-style reasoning. Finally, we conduct Sketch-Thinking Reinforcement Learning under supervision of SketchJudge to further generalize sketch-style reasoning ability. Experimental evaluation on four benchmarks reveals that our SketchThinker-R1 achieves over 64% reduction in reasoning token cost without compromising final answer accuracy. Qualitative analysis further shows that sketch-style reasoning focuses more on key cues during problem solving.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents SketchThinker-R1, a reinforcement learning framework designed to induce sketch-style reasoning in large multimodal models—defined as concise, goal-directed reasoning that retains essential logical steps while eliminating redundancy. The method is composed of three stages: (1) Sketch-Mode Cold Start, in which long-form reasoning traces from LLaVA-CoT-100K and Vision-R1-cold are systematically distilled into sketch-style sequences via GPT-5, followed by supervised fine-tuning to instill initial reasoning condensation capability; (2) SketchJudge Reward Model training, where an open-source LLM (Qwen2.5-7B-Instruct) is fine-tuned on paired reasoning samples (normal vs. sketch-style) to learn a reliable binary score for reasoning style, with no reliance on absolute length thresholds; and (3) Sketch-Thinking Reinforcement Learning, wherein the cold-started model is further optimized via GRPO, with rewards composed of accuracy (50%), format adherence (40%), and SketchJudge style score (10%). Evaluated across four standard multimodal benchmarks—MMMU, MathVision, VisuLogic, and PhyX—the method achieves an average reduction of 64.3% in reasoning token cost compared to the Vanilla-R1 baseline, with no degradation in answer accuracy and consistently superior efficiency.

### Strengths
1. This paper introduces a novel and comprehensive three-stage framework, SketchThinker-R1, which moves beyond superficial reasoning length constraints by directly fostering an "intrinsic sketch-style thinking" within large multimodal models. This approach represents a significant paradigm shift from external compression to internal cognitive efficiency.

2. The development and integration of the specialized SketchJudge reward model are a key strength. This model's ability to accurately evaluate "reasoning style" rather than merely length provides a precise and stable signal for reinforcement learning, enabling the model to learn context-aware reasoning condensation.

3. The framework consistently achieves substantial reductions in reasoning token cost (over 64% on average) across diverse benchmarks (MMMU, MathVision, VisuLogic, PhyX) while maintaining or even improving answer accuracy. This demonstrates a robust solution to the critical challenge of high computational inference costs in advanced multimodal models.

4. This paper benefits from clear writing, well-structured sections, and effective use of visual aids. Specifically, Figure 1 (quantitative results overview) and Figure 3 (qualitative reasoning examples) are particularly effective in conveying the core idea and the method's impact, making complex concepts accessible to the reader.

### Weaknesses
1. This paper relies on a powerful closed-source large language model for generating concise "sketch-style reasoning" data. It remains uncertain whether these generated data consistently encapsulate all necessary reasoning steps, which could potentially impact the interpretability of the subsequent black-box reasoning model and challenge the paper's assumption of mimicking human thought.

2. This paper claims to enhance efficiency without compromising accuracy; however, its accuracy baseline for comparison does not represent a model optimized for "absolute maximal accuracy." If a model, trained solely on the original long reasoning data (with consistent samples but without prior condensation), could achieve higher accuracy, then SketchThinker-R1 might be achieving its efficiency gains at a suboptimal accuracy ceiling, suggesting an unexplored trade-off between accuracy and efficiency.

### Questions
Considering the findings and conclusions presented in this paper, does this imply that we could embrace such condensed, sketch-style reasoning data from the pretraining phase to significantly reduce training costs?

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
This paper proposes SketchThinker-R1, a three-stage framework to train large multimodal models to produce “sketch-style” reasoning chains: first a cold-start stage where long form reasoning traces are converted into concise sketches and the model is fine-tuned on them; second a SketchJudge reward model is trained to explicitly score reasoning traces by favoring the sketch style; third a reinforcement-learning (RL) stage (using GRPO) encourages the model to produce sketch-style reasoning under the reward model’s supervision. Their motivation is that current large multimodal reasoning models use long chain-of-thought traces, which incur high token cost and increased latency, while human “sketching” uses fewer but more salient steps — thus they aim to reduce reasoning token cost without sacrificing accuracy. They build a dataset “SketchColdStart-20K” by converting 20K long-form reasoning traces (10 K from each of two multimodal reasoning datasets) into sketch‐style reasoning via a strong LLM (GPT-5) and then fine-tune the base model. They then sample a small RL set of 1 K from four domains (MMStar, MathVista, LogicVista, SeePhys) for the RL stage.

### Strengths
The motivation is good. Reducing token number is reasonable for multimodal tasks thinking. 

The three‐stage pipeline (cold‐start conversion of reasoning style, reward model, RL) is reasonably well‐designed and coherently described. The reward model is reasonable.

Experiments consistently show good results spanning different domains (visual reasoning, logic, physics) while showing substantial reductions in token cost while preserving good accuracy.

### Weaknesses
1. Suggest an experiment on ratio of the sketch reward. What if increasing the sketch coefficient and lower the format coefficient?

2. It is unclear if every baseline shares the same training data or just vanilla r1. Good to include more details.

3. How much time (GPU hour) in practice could be saved?

### Questions
See weakness.

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
The paper introduces SketchThinker-R1, an RL framework that trains multimodal LM to elicient concise reasoning instead of verbose ones, while keeping the same level of accuracy. The method contains three steps. The first step is use a strong LLM to converte existing long COT reasoning traces into short ones with numbered steps, and then do cold-start SFT. The second step is to train a SketchJudge reward model that can classify if a reasoning trace is "sketch style". The third step is do GRPO with the verifiable reward, format reward, and the sketchjudge reward.

The authors experiment this method on MMMU, MathVision, VisuLogic, PhyX, and show that this method can reduce reasoning tokens by >64% while keeping the accuracy.

### Strengths
1. The method gives good efficiency gains on multimodal reasoning, with 64% token reduction, while keeping the accuracy.

2. The method is simple and make sense. It shows that by carefully curate the SFT cold start data, and a new reward for sketch-style, it is possible to achieve similar reasoning process while greatly reduce the token count.

3. The ablations are well designed and provide good analysis.

### Weaknesses
1. The major weakness is that the scale of the experiments are very small. It only contains 1K prompts training set and 150 steps. It is questionable whether this method is scalable and generalizable on more tasks. What will happen if a bigger training set is used, and more FLOPs trained? Will it be continually improving, or this method is more unstable compared with vanilla R1? The paper would be stronger if include more scaling experiments.

2. More analysis on the reward models would make the paper stronger. Now the method uses a 0/1 reward on the "sketch" style. What if we use a dense reward? Also, it has a weight of 0.1. What will happen if we use other weights? Any reward hacking, or competing with other rewards observed during experiments?

### Questions
1. About the reward model SketchJudge. What alternatives have you tried? Like for example, a dense reward. Why using a binary reward in the end?

2. How is the weight 0.5, 0.4, 0.1 determined? What will happen if using other weights. Does the SketchJudge hurts accuracy if the weight is too big? Is there other method to combine these multiple objectives?

3. Have you run a human study on the interpretablity of the reasoning traces of the models you trained? Are the reasoning traces concise and reasonable, or they are just shorter but with many unreadable stuff?

### Soundness
2

### Presentation
4

### Contribution
3

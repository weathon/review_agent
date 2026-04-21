# Think Then React: Towards Unconstrained Action-to-Reaction Motion Generation

- Avg Score: 6.50
- Decision: Accept (Poster)
- Scores: 8, 6, 6, 6

## Abstract
Modeling human-like action-to-reaction generation has significant real-world applications, like human-robot interaction and games.
Despite recent advancements in single-person motion generation, it is still challenging to well handle action-to-reaction generation, due to the difficulty of directly predicting reaction from action sequence without prompts, and the absence of a unified representation that effectively encodes multi-person motion.
To address these challenges, we introduce Think-Then-React (TTR), a large language-model-based framework designed to generate human-like reactions.
First, with our fine-grained multimodal training strategy, TTR is capable to unify two processes during inference: a thinking process that explicitly infers action intentions and reasons corresponding reaction description, which serve as semantic prompts, and a reacting process that predicts reactions based on input action and the inferred semantic prompts.
Second, to effectively represent multi-person motion in language models, we propose a unified motion tokenizer by decoupling egocentric pose and absolute space features, which effectively represents action and reaction motion with same encoding.
Extensive experiments demonstrate that TTR outperforms existing baselines, achieving significant improvements in evaluation metrics, such as reducing FID from 3.988 to 1.942.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper present a new method for generating a reaction motion based on an action motion in a two persons interaction. The proposed architecture uses LLMs to understand the input action motion and then infer possible reactions to this action. The action motion is first converted into tokens that are used by the LLM to infer the type of action and predicts the reaction and the corresponding token that are then projected back into motion space. The method uses several training schemes to achieve a better understanding of the action and generate a better reaction. The method outperform the state of the art quantitatively on the Inter-X dataset.

### Strengths
- The idea of splitting the task of reaction generation between a "thinking" process using LLM and a "reacting" process using VQ-VAE is very interesting. This method will probably be able to scale well as new human interaction datasets are released.
- The architecture and training scheme as well as the intuitions behind them are sound.
- Treating the global pose with a separate tokenizer is a good idea and allow for more flexibility while staying accurate.
- "re-thinking" is also interesting to correct the generation in an online manner.
- Method beats the state of the art by a good margin on most metrics.
- Extensive ablation on the pretraining.

### Weaknesses
- Very few qualitative results. There is no qualitative comparison against the state of the art and no videos to show the actual motions. the lack of qualitative results makes it very difficult to evaluate the model qualitatively and to see if quantitative improvements correspond to similar qualitative improvements.
- The "pose tokenizer" described in the main paper (3.2.1) is different from the one described in the appendix. Is the tokenizer from the appendix the "space tokenizer" ? This should be clarified.
- Sec 3.3.1 (3) "we take the first half of action and the second half of reaction sequence of tokens as input". Is the opposite operation also performed ? i.e. "take the SECOND half of action and the FIRST half of reaction sequence of tokens as input" ? The text says "and vice versa" but it is not clear what it refers to.
- The authors do not explain why they use only the 2D positions for the "absolute pose" and not the 3D positions.
- There are some issues with notations:
    - line 204 "k" is never defined and why take "k-l" as input ?
    - line 231 "M" is never defined.
    - line 231 why change the notation from "Nf" to "L" for the number of frame ?
    - line 266 why use "t" for timestep when it was defined as "l" (or maybe "k") on line 204 ?
- Some of the writing could use improvement:
    - lines 203-206 : phrasing is very confusing and makes it difficult to understand what the authors want to say.
    - lines 266-268 : same as above
    - The way "pose" and "motion" have been defined and are then used is sometime confusing.

### Questions
- Qualitative comparison are very important in these type of works. Including them is always a good idea as they help in showing the performance of the method. Also since the work is on motion video qualitative results are always appreciated.
- Typo line 231 : "number of frames or original motion"

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
The paper presents Think-Then-React, a novel method for action-to-reaction motion generation. It introduces a unified motion encoder that tokenizes the initial position and subsequent poses. Besides it employs a large language model for understanding and generating both motion and text. The model is pre-trained on various tasks, therefore it enhances its performance in generating coherent reactions based on observed actions.

### Strengths
- The paper aims at improve the quality of action-to-reaction generation, which is an important issue.
- The paper introduces a unified space-pose token representation and proves its effectiveness.
- The significant improvement in performance compared to the baseline is a major advantage.

### Weaknesses
- The training data utilized a mixture of two datasets, which may lead to an unfair comparison.
- The experiments only presented numerical metrics on the test set and did not include evaluations of the generated actions by real humans.

### Questions
- I am confused by the dataset used for training. Do other methods also use mixed dataset? If they only use one dataset but you use the mixture of two, it will cause unfairness. Please provide the information about the dataset used by other methods in the paper and the details about dataset mixture strategies.

- Does your paper include a comparison involving evaluations of the generated actions by real humans (i.e., a user study)? In the generation field, the importance of user study results is comparable to that of numerical metrics.

### Soundness
3

### Presentation
3

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
The author trains an online complicated framework to generating reaction motion based on the given action motions.  Firstly, the author introduces a unified space-pose representation that effectively converts both absolute space and egocentric pose features into discrete tokens, facilitating LLM training. In addition, the authors introduce the think-then-react framework to get better results.

### Strengths
1.  The figure and render visualization is good. 
2. The authors introduce a unified space-pose representation that effectively converts both absolute space and egocentric pose features into discrete tokens, facilitating LLM training.

### Weaknesses
1. Poor writing. This paper is hard to follow. It's hard for me to distinguish the differences between special words, like “motion”, "pose", "action", "action motion", etc. 
2. The training process is complicated. Besides the pertaining and fine-tuning, to improve the training stability, the author also needs to check the validation loss manually and switch the training source.

### Questions
1. Could the author provide more video visualization to show the results? It's tough for me to see the generation results from images.
2. Could the author explain further why the thinking part works? i.e. why we need the text as a bridge to connect the input action and output action? From my general understanding, the translation from input action to text has already introduced the information loss. 
3. Did 1 FPS and 0.5-second time interval really work? This may lead to heavy leggy and jitter. Can 0.5 seconds really work for an online system?
4. Since the performance depends on the text generation process, could the author try some motion caption benchmarks?

### Soundness
2

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
4

### Summary
The submission handled the problem of human reaction generation. Specifically, the submission proposed to utilize a unified motion token representation encompassing both egocentric and global spaces. Then, an LLM-based online reaction generation framework TTR is proposed to leverage textual information for training. TTR involves action intention inference as the first step and precise reaction token prediction as the second step. Impressive quantitative improvements are shown.

### Strengths
- Explicit decomposition of egocentric poses and global transformations is reasonable, especially considering human reactions.

- Introducing explicit intention modeling as an intermediate stage for reaction generation has been proven helpful.

### Weaknesses
- The overall writing clarity could be improved. Please refer to Questions.

- The evaluation settings could be ambiguous. Please refer to Questions.

- The effectiveness of the proposed unified motion tokenizer is not specifically discussed in experiments. Is it feasible to treat TTR w/o ALL P.T. as MotionGPT w/ unified representation? To understand this, details on how MotionGPT is adapted to multi-person scenarios should be provided.

- Despite the considerable quantitative improvements, the improved quantitative metrics are essentially indirect proxies of the real reaction generalization performance. A qualitative comparison of how the proposed method outperforms previous SOTAs is still expected. Animated motions would be a better form of visualization.

- Moreover, a failure case study is also expected if possible.

### Questions
- In L219, it is unclear how a pair of persons should face the positive z-axis. It is more understandable that the reactee faces the positive z-axis.

- In L221, are (x,z) corresponding to the pelvis 2D coordinates? 

- Terms in L231 are inconsistent, making it hard to understand.

- Why the evaluation model from Inter-X is not adopted, which has a high ACC. over 90% compared to the 46.3% of the adopted model?

### Soundness
3

### Presentation
2

### Contribution
3

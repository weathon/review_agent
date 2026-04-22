# Benchmarking MLLM-based Web Understanding: Reasoning, Robustness and Safety

- Avg Score: 2.50
- Decision: Reject
- Scores: 4, 2, 2, 2

## Abstract
Multimodal large language models (MLLMs) are increasingly positioned as AI collaborators for building complex web-related applications like GUI agents and front-end code generation. However, existing benchmarks largely emphasize visual perception or UI code generation, showing insufficient evaluation on the reasoning, robustness and safety capability required for end-to-end web applications.  To bridge the gap, we introduce a comprehensive web understanding benchmark, named WebRSSBench, that jointly evaluates Reasoning, Robustness, and Safety across eight tasks, such as position relationship reasoning, color robustness, and safety critical detection, etc. The benchmark is constructed from 729 websites and contains 3799 question–answer pairs that probe multi-step inference over page structure, text, widgets, and safety-critical interactions. To ensure reliable measurement, we adopt standardized prompts, deterministic evaluation scripts, and multi-stage quality control combining automatic checks with targeted human verification. We evaluate 12 MLLMs on WebRSSBench. The results reveal significant gaps: models still struggle with compositional and cross-element reasoning over realistic layouts, show limited robustness when facing perturbations in user interfaces and content such as layout rearrangements or visual style shifts, and are rather conservative in recognizing and avoiding safety critical or irreversible actions. Our code is available at https://anonymous.4open.science/r/WebRSSBench/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Propose WebRSSBench, evaluating the web understanding ability of MLLM from three dimensions: Reasoning / Robustness / Safety, covering 8 subtasks. It covers 729 websites and 3,799 QA.

### Strengths
- The coverage and issue selection are reasonable, put the urgently needed capabilities of actual web page agents (spatial relationships, cross-element semantics, form/prompts, key button recognition, and risk avoidance) in a continuous evaluation, with clear engineering value.

### Weaknesses
- The text distortion algorithm includes reversing strings, shuffling characters, etc. (Appendix C.2), which directly destroys readability and usability, going beyond the premise of "semantic preservation". In this case, measuring "robustness" is no longer fair?

- In color robustness and text/form semantic tasks, the authors use cross-model consensus or "semantic centroid" as the gold standard, which embeds the preferences of closed-source models into the gold standard, and weakens reproducibility?

### Questions
see weekness above

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces WebRSSBench, a benchmark designed to evaluate three key aspects of multimodal large language models (MLLMs): reasoning, robustness, and safety. The benchmark includes eight tasks across 729 websites and 3,799 QA samples. Data are collected from Mind2Web, WebMMU, WebSRC, and design-oriented webpages, with a subset selected based on specific criteria. The authors show that closed-source models generally outperform open-source models and demonstrate that LoRA fine-tuning helps narrow this performance gap.

### Strengths
- The paper defines interesting and practical tasks, such as hint text prediction and form filling, which could be valuable for assessing whether MLLMs truly understand webpage content.

- The inclusion of a safety evaluation dimension is notable and timely, given the increasing deployment of autonomous agents.

- State-of-the-art closed-source MLLMs are included in the evaluation, providing a strong comparative baseline.

### Weaknesses
- The description of how ground truth is derived is unclear. As understood, the ground truth is based on the consensus of the 12 evaluated models. If a new model (e.g., model #13) were to be tested, would the evaluation require recomputing the consensus with all 13 models, potentially altering previous results? More context and clarification are needed here.

- Table 1 appears cluttered; splitting it into two tables or moving less critical details to the appendix would improve readability.

- The main text omits essential details of the fine-tuning process. It is unclear why LoRA was chosen over full fine-tuning and what data were used for training. Appendix D mentions LoRA settings and an 8:1:1 train/validation/test split, but the data source remains unspecified. Are the authors fine-tuning on 80% of the benchmark’s own data and evaluating on the remaining 10%?

- Missing citations: prior work on MLLM robustness to positional and visual perturbations—such as PairBench [1] for spatial reasoning and color perturbation robustness, and VisMin [2] for minimal-change spatial reasoning—should be discussed to better situate this benchmark within existing research.

- Numerous minor typos, including inconsistent capitalization (e.g., “Model” in L335, “closed Source Model” in L405, “Model” in L465, and “We” in L243).

[1] PairBench: Are Vision-Language Models Reliable at Comparing What They See?

[2] VisMin: Visual Minimal-Change Understanding

### Questions
- How is the ground truth maintained or extended when new models are added to the benchmark?

- What do the authors exactly mean by "extensibility" and how can WebRSSBench be applied to new test cases and dimensions? 

- What data are used for LoRA fine-tuning, and how are they separated from evaluation data to prevent data leakage?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The authors propose WebRSSBench, a benchmark for website understanding that combines reasoning, robustness, and safety across 8 tasks built from multiple sources (729 sites). They evaluate 12 models and the main findings is that reasoning is hard, models are brittle to UI perturbations, and closed-source models lead.

### Strengths
- Like the clear task coverage spanning spatial reasoning (e.g. relative position), UI grouping, form filling, hint text plus three perturbation types (color, text, layout) and a safety task.
-  Dataset sources draw from real webpages/design communities and prior datasets (Mind2Web, WebMMU, WebSRC, Moz Top 500 etc).
- Clear qualitative takeaways on reasoning difficulty and brittleness to layout/color/text changes.

### Weaknesses
- Several ground truths are derived from cross-model consensus (e.g., majority-voted buttons / semantic centroids), risking circularity and shared bias. Please consider human-verified GT or task-verifiable signals.
- not an expert, but here safety is evaluated via "does the page contain any safety-critical button? output its text, else ‘sorry’" and recall is used because pages are curated to contain such elements. But the paper does not describe a gold process for identifying safety critical affordances (beyond manual screening) nor tackle ambiguity (e.g., “reset”, “archive”).
- Major concern to me is perturbations may not mirror real breakages, feels too synthetic: recoloring 10–30% of buttons with strong colors is synthetic, but real failures often come from contrast ratios, disabled states, or theme toggles. Consider WCAG driven contrast shifts and "disabled/ghost" button styles. character swaps , reversal, etc. feel adversarial for OCR, but production issues are often localization,  icon-only buttons, or mixed scripts. Include these... same for DOMs modern systems are updated.
- Filtering (e.g. page length thresholds) may bias toward mid-complex pages, nice to quantify how filtering shifts distributions (language, verticals, accessibility)
- Color robustness score for example is  tied to performance gaps rather than standard stat measures; no motivation of the 20 bound, and no uncertainty intervals. 
- LoRA gains are promising but under specified: splits, leakage controls (site/template level), seeds, compute, early-stopping and transfer across perturbations are missing.

Consider including one or more of the above or provide evidence if any of the points are not meaningful in the current study.

- [minor] avoid hyphenation and line breaks in title.
- Table 2 header typo: "Positon" instead of  “Position”.
- Prompt typo: "botton" instead of "button"

Suggestions:
- Expand perturbations: ark-mode/contrast changes, disabled/ghost buttons, icons only etc..
- add human correlation studies

OVerall needs substantial revision!

### Questions
- How often do consensus derived groundtruth disagree with human annotations on a held out set? would be nice to see a correlation.
- [optional] In safety detection, how do you differentiate critical vs. cautionary actions (e.g. "Sign out" vs. "Delete account")?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper introduces WebRSSBench, a benchmark for multimodal LLMs that jointly evaluates Reasoning, Robustness, and Safety across eight web-understanding tasks. It is made of 729 real webpages and 3799 QA pairs. It provides deterministic scripts and evaluates 12 open- and closed-source MLLMs. Overall, results show that closed-source models have better performance, in particular on safety. Authors also surface three brittleness patterns under color, text, and layout perturbations. Finally, targeted LoRA finetuning was experimented to demonstrate boost in position reasoning, UI grouping, and color robustness.

### Strengths
The paper presents diverse corpus with 729 webpages and 3799 QA pairs drawn from real sites and design communities. 

This paper introduces programmatic color, text, and layout perturbations with before, after comparisons to make robustness measurement explicit and reproducible.

The use of deterministic scripts and multistage quality check in improving reliability of results ensures the high standard of the dataset.

Author has also trained with LoRA finetuning and demonstrates improvement in the needed areas.

### Weaknesses
The author claims safety evaluation as a strength of the benchmark. However, there are only 45 questions under the safety category. It would be much more statistically sound if the author could include more questions in this category. 

It would be helpful to provide more reasoning around why these scenarios are selected: position relationship reasoning, form filling, hint text prediction, and UI grouping. And if they have covered MLLM’s usage scenarios to a good extent. So that testing on these tasks can accurately reflect models’ abilities. 

There is also a large variation in the number of questions between each category, ranging from 720 to 45, it is best to have them in closer numbers. 

Provide more details in training, how are training data constructed etc. 

It would be great to see how human perform as a baseline.

### Questions
As listed above.

### Soundness
3

### Presentation
3

### Contribution
2

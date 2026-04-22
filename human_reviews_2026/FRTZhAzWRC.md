# WildFeedback: Aligning LLMs With In-situ User Interactions And Feedback

- Avg Score: 3.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 2, 2, 4, 6

## Abstract
As large language models (LLMs) continue to advance, aligning these models with human preferences has emerged as a critical challenge. Traditional alignment methods, relying on human or LLM annotated datasets, are limited by their resource-intensive nature, inherent subjectivity, misalignment with real-world user preferences, and the risk of feedback loops that amplify model biases. To overcome these limitations, we introduce WildFeedback, a novel framework that leverages in-situ user feedback during conversations with LLMs to create preference datasets automatically. Given a corpus of multi-turn user-LLM conversation, WildFeedback identifies and classifies user feedback to LLM responses between conversation turns. The user feedback is then used to create examples of preferred and dispreferred responses according to users' preference. Our experiments demonstrate that LLMs fine-tuned on WildFeedback dataset exhibit significantly improved alignment with user preferences, as evidenced by both traditional benchmarks and our proposed checklist-guided evaluation. By incorporating in-situ feedback from actual users, WildFeedback addresses the scalability, subjectivity, and bias challenges that plague existing approaches, marking a significant step toward developing LLMs that are more responsive to the diverse and evolving needs of their users.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper proposes WildFeedback, a novel framework that leverages in-situ user feedback to automatically construct preference datasets. These datasets are then used to generate preferred/dispreferred pairs for preference tuning of LLMs. Experimental results demonstrate the potential effectiveness of the proposed approach.

### Strengths
*  The paper tackles a timely and important problem: how to automatically mine user preferences without relying on explicit up/down votes on LLM responses.
*  The proposed framework aims to overcome the limitations of both synthetic and human-annotated preference datasets, which is a meaningful and practically relevant direction.

### Weaknesses
*  Lack of fine-grained methodological differentiation from concurrent work such as UltraFeedback. While UltraFeedback uses GPT-4 to generate rankings and derive preferences, the proposed method also relies on GPT-4 for preference determination. Despite architectural or procedural differences, it is unclear whether this work provides a fundamentally distinct contribution beyond UltraFeedback.
*  Potential evaluation leakage (“double dipping”): The checklist-guided evaluation relies on LLMs both to generate preference data and to summarize user preferences which is then used to align LLM evaluation. This design may inadvertently bias results toward the model’s own judgments, undermining the objectivity of the evaluation.
*  Experimental results are not consistently supportive of the claimed significant and consistent enhancement. For instance, in Table 3, UltraFeedback outperforms WildFeedback on many occasions.

### Questions
1.  Lines 191–192: Is it correct that $\kappa$ = 0.5 is considered low agreement and $\kappa$ = 0.69 is considered moderate agreement? Please clarify your interpretation of Cohen’s κ thresholds.
1.  Line 268: The claim that 57.14% and 63.27% are “similar” seems questionable. For a binary agree/disagree task, random performance is 50%; thus, 57.14% is only 7% above random, while 63.27% is 13% above random—nearly double that margin.

### Soundness
1

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
The authors proposed WildFeedback, an automatic preference data generation framework by identifying user satisfaction signals in multi-turn conversations. Models trained with WildFeedback dataset outperforms those trained with UltraFeedback across Phi3, Llama3 and Qwen2. A check list guided evaluation procedure is also introduced to mitigate the mismatch between annotators’ preferences and actual user preferences

### Strengths
* WildFeedBack outperformed UltraFeedBack on multiple open source LLMs, demonstrating robust improvement.
* The automatic preference data construction framework provides easy implementation.
* The writing is clear and easy to follow.

### Weaknesses
The overall novelty of this work is limited. On implementation side, WildFeedback proposed effective automatic pipeline to identify user satisfaction and generate human preference data. But such pipeline appears to be a task-specific solution to this scenario combining existing techniques. The theoretical and heuristic insights from the work is also limited.

### Questions
On Phi3 and LlaMA3, UF seems to have higher win-rate if not length-controlled. Would the authors provide some qualitative generation from UF trained model with and without length control?

### Soundness
3

### Presentation
3

### Contribution
1

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
**Summary**

The paper proposes aligning large language models (LLMs) using **in-situ annotations**—that is, satisfaction (SAT) and dissatisfaction (DSAT) cues, along with user edits extracted from real user–assistant conversations. These cues are converted into **instance-level preference summaries** (“checklists”), which are then used to form **preferred/dispreferred pairs** (generated either by GPT-4 or on-policy). The models are fine-tuned using **SFT** followed by **DPO**, and the same checklists are later reused to guide the evaluation process for more reliable judgments.
Specifically, the framework first identifies utterances in multi-turn dialogues that match predefined rubric cues (e.g., SAT: “thank you”; DSAT: “revise it,”).
For each detected case, it extracts the conversation up to the model response that triggered the DSAT signal as the **prompt**, treating that response as the **dispreferred output**. The user’s preferences from the feedback are summarized into a **checklist**.
Finally, the preferred response is generated under the guidance of this checklist (either by an expert model or by the same policy model), with a safety instruction and moderation applied. The same checklist is provided to the judge during evaluation to better distinguish preferred from dispreferred responses.

### Strengths
**Strengths**
* **Grounding in real interactions:** Builds preference data directly from real multi-turn conversations (WildChat).

* **Clear and reproducible training design:** 

* **Consistent performance gains:** Shows improvements on AlpacaEval 2, Arena-Hard, and MT-Bench benchmarks across multiple backbones (e.g., Phi-3 LC win-rate +10.6 points on AlpacaEval 2).

* **Human validation:** Validates both SAT/DSAT detection and checklist-guided evaluation with moderate human–model agreement.

### Weaknesses
**Weaknesses**

1. **Noisy in-situ supervision.**
   SAT/DSAT labels are auto-classified by GPT-4; human validation shows only **moderate** agreement (κ≈0.69 SAT / 0.50 DSAT), so non-trivial label noise can propagate into pair construction and training.  

2. **Single cues are insufficient as ground truth.**
   Individual cues (e.g., “thank you”, “revise it”) may not capture full intent; even with checklist summaries, relying on in-situ cues alone risks ambiguity. (They also rely on checklists during evaluation.)  

3. **Limited methodological novelty & strong LLM dependence.**
   Core pieces (in-situ cues, SFT→DPO, LLM-as-judge) are known; the framework’s novelty is mainly integration + checklists. GPT-4 is used to generate preferred answers (WF-GPT-4) **and** to judge results, inviting circularity.    

4. **No variance/seed reporting.**
   Results are reported as single win/tie/lose or scores without error bars; seeds/repeats aren’t specified—please report multi-seed means with CIs/SDs.  

---

**Overall assessment (concise)**
A **practical** pipeline with **consistent gains** on AlpacaEval 2, Arena-Hard, and MT-Bench, but **incremental novelty** and **heavy dependence on GPT-4 as judge** remain concerns.

### Questions
### Q1. Reliability of in-situ SAT/DSAT

 How do you verify cues reflect genuine satisfaction/dissatisfaction (not politeness)?

### Q2. Robustness to noisy labels

 How sensitive are results to SAT/DSAT noise? Any threshold where gains collapse?

**Request:** **Noise-level ablation**: flip p% labels for p∈{5,10,20,30} (random), report **curves with ≥2 seeds (mean±SD/CI)** 

### Q3. What’s new vs ULTRAFEEDBACK, SPUR, WILDBENCH?

 Precisely distinguish your contribution from prior work; what’s beyond combination ?

**Request:** A **comparison table** (data source, supervision unit, role in training/eval, on-policy vs GPT-4, curation). **Ablations:** (i) train **without checklists**; (ii) train on **ULTRA** but **evaluate with your checklists** (and vice versa); (iii) replace your judge with WILDBENCH default (no checklists) and report deltas.

### Q4. Cross-judge (LLM & human) evaluation
 Do gains hold across judges and prompt settings?

**Request:** Judges: **Claude**, **Llama-3-70B** (or comparable).

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
This work presents a method for synthesizing preference data from real user-llm multiturn interactions. Their method relies on first automatically identifying instances where users provide feedback for the LLM's response in real user-llm dialogues (sourced from wildchat). The feedback is then used to generate an alternative response to construct the synthetic preference pair. The authors then demonstrate that training LLMs using these synthetic preference pairs demonstrates gains across a variety of standard LLM chat benchmarks when compared against standard methods of generating synthetic preference data (i.e., UltraFeedback).

### Strengths
1. This work has great presentation, including the appendix which is quite thorough. 

2. The method itself is straightforward and demonstrates consistent gains across a number of benchmarks and base models.

### Weaknesses
1. While this work uses some established benchmarks, they alter the evaluation by introducing their own checklist-based judge. While WildBench employed a similar method for their evaluations, their checklists generation method and validation was significantly more extensive to ensure that it is comprehensive, accurate, and unbiased toward particular LMs. Including the including the performance as measured by the original benchmark prompts and settings in addition to evaluating on Wildbench would improve the validity of the experiments.

2. While the methods and settings are distinct, there is enough similarity to [1] to warrant discussion or even direct comparison in evaluations. Overall, the related work section on Feedback Learning for LLMs could be more though, as there is a significant amount of literature on methods for learning from implicit signals from real user data (e.g., [2], [3]). Including [4] as a baseline could also make sense.

[1] User Feedback in Human-LLM Dialogues: A Lens to Understand Users But Noisy as a Learning Signal
Yuhan Liu, Michael J.Q. Zhang, Eunsol Choi
EMNLP 2025

[2] Leveraging Implicit Feedback from Deployment Data in Dialogue
Richard Yuanzhe Pang, Stephen Roller, Kyunghyun Cho, He He, Jason Weston
EACL 2024

[3] Retrospective Learning from Interactions
Zizhao Chen, Mustafa Omer Gul, Yiwei Chen, Gloria Geng, Anne Wu, Yoav Artzi
ACL 2025

[4] KTO: Model Alignment as Prospect Theoretic Optimization
Kawin Ethayarajh, Winnie Xu, Niklas Muennighoff, Dan Jurafsky, Douwe Kiela
ICML 2024

### Questions
What do the generated examples look like? Can you provide statistics like length for the synthesized responses? While this is a synthetic dataset generation method, it would also be worthwhile to include samples and such statistics from the dataset as if it were a standard dataset paper.

### Soundness
3

### Presentation
4

### Contribution
3

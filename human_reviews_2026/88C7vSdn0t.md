# Time-To-Inconsistency: A Survival Analysis of Large Language Model Robustness to Adversarial Attacks

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 8, 2, 6, 4

## Abstract
Large Language Models (LLMs) have revolutionized conversational AI, yet their robustness in extended multi-turn dialogues remains poorly understood. Existing evaluation frameworks focus on static benchmarks and single-turn assessments, failing to capture the temporal dynamics of conversational degradation that characterize real-world interactions. In this work, we present a large-scale survival analysis of conversational robustness, modeling failure as a time-to-event process over 36,951 turns from 9 state-of-the-art LLMs on the MT-Consistency benchmark. Our framework combines Cox proportional hazards, Accelerated Failure Time (AFT), and Random Survival Forest models with simple semantic drift features. We find that abrupt prompt-to-prompt semantic drift sharply increases the hazard of inconsistency, whereas cumulative drift is counterintuitively \emph{protective}, suggesting adaptation in conversations that survive multiple shifts. AFT models with model–drift interactions achieve the best combination of discrimination and calibration, and proportional hazards checks reveal systematic violations for key drift covariates, explaining the limitations of Cox-style modeling in this setting. Finally, we show that a lightweight AFT model can be turned into a turn-level risk monitor that flags most failing conversations several turns before the first inconsistent answer while keeping false alerts modest. These results establish survival analysis as a powerful paradigm for evaluating multi-turn robustness and for designing practical safeguards for conversational AI systems.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
The paper models multi-turn LLM failures as a survival analysis modeling problem. With this, the authors demonstrate that the rate of failure within multi-turn LLM interactions are not constant hazard models, and that semantic drift is a strong predictor for failure.

### Strengths
I think the paper tackles an important consideration: as AI assistants are used more and more for long periods of time, how well do they remain consistent, especially under adversarial pressure. As far as I am aware this is high in originality.

The paper is clear, and well-written. The paper also is strong in general scholarship, suitably referencing existing literature appropriately, and nicely situates their work within the broader body of literature on AI evaluation, consistency, and survival analysis. 

The experimental set-up is strong, with a range of metrics, survival models, and LLMs. The results shed light on the fact that the proportional hazards assumption doesn't necessarily hold across important features. This is a significant finding, particularly if further validated on additional datasets.

### Weaknesses
The limit of turn interactions to 8 steps is strangely limiting --- current models can engage with conversations over significantly larger context windows and more steps. If there are technical reasons for sticking to this -- e.g., if it were a facet of the dataset used or something similar, this should be explicitly mentioned in the paper. 

The paper should explore a wider range of models. I think capturing a broader range of model sizes would be insightful because the survival times could be explored in the context of "scaling laws". I think the paper's analysis would likely explain newer models' ability to complete longer tasks (as identified by METR https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/). Having a robust model explain this behaviour would be valuable. 

The paper proposes that real-time monitoring of conversational drift could be employed to keep LLMs on track. This statement, while I think is likely, needs some empirical validation which the paper doesn't provide. That is, a demonstration of real-time monitoring would go a long way to validate the claim00

### Questions
Can you expand on and demonstrate the use of real-time monitoring for keeping LLMs consistent and on track? 
What is the purpose of limiting the number of turn interactions to 8 steps?
How would you anticipate these time-to-inconsistent results scale with increased model size / parameters?

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
2

### Summary
This paper applies survival analysis to multi-turn LLM conversations, measuring the time (in turns) until the LLM produces a wrong answer (failure).

**Method**: The authors first construct a feature vector for every turn in a multi-turn conversation, and then fit three classes of survival analysis models to those features (two Cox proportional hazards (PH) models, four accelerated failure time (AFT) models, and a random survival forest). Failure is defined as the LLM giving an incorrect answer.

Using sentence-transformer embeddings of prompts, the features for each conversation turn are
1. prompt-to-prompt drift: cosine dissimilarity between the current and previous prompt embedding
2. context-to-prompt drift: cosine dissimilarity between the current and the (mean) previous prompt embeddings
3. cumulative drift: the sum of prompt-to-prompt drifts up to the current turn
4. discrete covariates: domain of the conversation topic, the question difficulty, and the model/model family
To the best of my understanding, this only considers *user* prompts; the features do *not* consider LLM responses.

**Experiments**: The authors apply all methods to the MT-Consistency benchmark, which contains multi-turn conversations for 9 LLMs. The experiments only use conversations with an initially correct answer (hence measuring if the LLM can be "swayed") and split those conversations into a 80%-20% train-test split.

The paper then evaluates how well different survival analysis models predict failure. AFT models are overall more accurate and better calibrated, likely because the proportional hazards assumption of PH models is violated for crucial features. The authors use the proportional hazards model to analyze their features; they find that drift between subsequent conversation turns increase the risk most, while the cumulative drift over a conversation (e.g., a gradual topic change) reduces the risk (in that PH model).

**Takeaways**: An abrupt change in semantics (in terms of prompt embedding similarity) increases the risk of LLMs providing a wrong answer, while a gradual change over many terms can actually reduce the risk. In particular, this implies that changes in topic are not detrimental (but can be good) as long as they are gradual. The risk of failure is not constant over turns, but changes as a conversation progresses. Lastly, the authors propose to use a lightweight AFT models in production to detect and intervene prompts where the risk of failure dramatically increases.

### Strengths
1. The results hint that lightweight AFT models could be a simple additional safeguard for practical LLM deployments. I'm curious to see if their efficacy holds up in an experiment.
2. The author's core motivation, moving away from single-turn evaluations or static metrics for multi-turn conversations, is clear and reasonable. The related work section provides a good contextualization of this motivation.
3. Similarly, the general idea of evaluating multi-turn LLM conversations from the perspective of survival analysis is novel and interesting.
4. The method description is very self-contained and detailed. Thus, the paper is also accessible to readers that might be less familiar with survival analysis.

### Weaknesses
While I'm not an expert in survival analysis, I could understand the overall methods in the paper. However, the paper could greatly benefit from a systematic overhaul of the presentation; I found it challenging to understand what the actual goals and contributions are, and I am still unsure about key parts of the feature engineering. In addition, the paper's takeaways are relatively narrow and could benefit from more discussion or experiments.

**Paper presentation**: The paper's overall goal/contribution could be presented more clearly. The abstract and introduction led me to believe that the paper's primary aim is to *analyze LLMs in multi-turn conversations* using survival analysis (e.g., L17-18, L58-59). However, the bulk of the paper focuses on *designing survival analysis models*, and the experiments evaluate *those methods* (not LLMs). In fact, only the last two pages contain any results for LLMs, without focusing on individual architectures or families, and the main takeaways and contributions became only clear to me when reading the discussion. Crucially, the key takeaway that survival analysis could be used as a practical safeguard, is never mentioned until the last paragraph of the discussion. The paper would greatly benefit from mentioning the actual goal and contributions early, so that readers understand how to interpret the method and results.

**Confusion around feature engineering**: Potentially due to the presentation issues, I found the first three features in Section 3.2 (Equations 1-3) difficult to understand. In particular, the writing should clarify whether drift uses the embeddings of *user prompts*, *LLM completions*, or a *concatenation* of the two. I.e., the paper should make it clear whether survival analysis is done w.r.t. to user inputs or LLM outputs; this is never mentioned explicitly. Adding to this confusion is the phrasing in Section 3.2: $e_t$ is multiple times defined as the *user* prompt embedding (L146, L152), but Prompt-to-Prompt drift mentions the shift between *conversational turns* (L149-150), which could entail a user prompt followed by an LLM response or vice versa. Similarly, $\mathbf{\bar{e}_{1:t-1}}$ is both referenced as the mean embedding of all previous *prompts* (L147-148), but then later seems to entail the full conversation context (which I initially presumed to include the LLM's responses) on L157.
Clarity about what the features entail is very important in my opinion, as all of the paper's insights relate around the semantic dissimilarity of user/LLM prompts. In my initial read, I understood the features to be about the concatenation of user prompts and LLM responses; after studying the paper more, I now believe features only consider user prompts, but I am still unsure.

**Significance**: The paper's contributions in their current form are a bit limited and could be expanded through more discussion or experiments. First, while having evidence for it is nice, the insight that an abrupt change in the conversation is more likely to induce LLM failure is in itself not surprising. Second, the finding that a gradual change in semantics might help robustness is surprising, but could benefit from more depth and additional evidence beyond the hazard ratio of a PH model (e.g., a controlled study). Lastly, the main practical takeaway in my opinion (and ultimately the reason to apply survival analysis in the first place) is that AFT models could serve as safeguards. However, this is only mentioned in one paragraph (L464-471); I would find additional experiments and depth on this very exciting.

### Questions
1. Are the sentence-transformer embeddings in Section 3.2 only taken over user prompts? Or are they calculated differently?
2. Section 5.4 and Figure 1 show that Prompt-to-Prompt drift is an important feature for failure prediction. Similarly, the fact that cumulative drifts are protective (one of the main insights of this paper) seem to stem from the same analysis. However, this is done w.r.t. a PH model, and the PH assumption seems not to hold for Prompt-to-Prompt drift (Table 2). How accurate are the insights given the assumption mismatch?
3. The results in Section 5.1/Tables 1 and 2 mention model-drift interaction terms for AFT models, but Section 3.3 only mentions such a term for the PH models. What is the interaction term for AFT models?

### Soundness
3

### Presentation
1

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper applies survival analysis to model when and why LLMs fail during multi-turn conversations. The key finding: abrupt prompt-to-prompt semantic shifts increase failure risk by 2-4.6, while gradual cumulative drift is protective (reduces risk). This means the velocity of conversational change matters more than total distance traveled—models can adapt to gradual topic evolution but break under sudden semantic jumps. Accelerated Failure Time models achieve 87.4% accuracy in predicting failures with 48% better calibration than baselines, enabling real-time monitoring systems that detect dangerous conversational shocks before failure occurs.

### Strengths
Reframes multi-turn robustness as a time-to-event problem and is (to my knowledge) the first comprehensive use of survival analysis (Cox/AFT/RSF) for LLM dialogue failure.

Introduces semantic drift features (prompt-to-prompt, context-to-prompt, cumulative) as time-varying covariates—an inventive, interpretable way to capture conversational dynamics.

Rigorous 36,951-turn analysis across 9 models. Multi-paradigm approach (Cox, AFT, RSF) ensures robust conclusions.

Establishes survival analysis as a powerful paradigm for evaluating conversational robustness, likely to influence future benchmarks and tooling.

Produces actionable design guidance (detect/mitigate abrupt drift; use AFT-style monitors for real-time risk stratification).

### Weaknesses
Entire analysis uses MT-Consistency benchmark with one adversarial protocol. 

Paper doesn't distinguish why models fail—sycophancy vs. context confusion vs. knowledge gaps all labeled as "failure."

### Questions
Your correlation between abrupt P2P drift and failure is compelling, but could high P2P simply indicate harder follow-up questions rather than causing failure?

Why does cumulative drift protect models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work presents the first large-scale survival analysis of conversational AI robustness, modeling failure as a time-to-event process across 36,951 conversation turns and nine LLMs.

The study reveals that abrupt semantic drift sharply increases failure risk, whereas gradual, cumulative drift helps maintain longer, more stable dialogues.

These findings establish survival analysis as a powerful new framework for evaluating LLM robustness and guiding the design of resilient conversational agents.

### Strengths
1. This paper proposes a multi-turn assessments for LLM robustness.

### Weaknesses
1. ***Missing Conceptual Depth***
- The introduction states the idea of using survival analysis but doesn’t explain why this framework is particularly appropriate or transformative for conversational robustness.

- It should elaborate how “time-to-failure” parallels conversational degradation (e.g., semantic drift, confidence decay, compounding reasoning errors).

- Suggest adding 1–2 sentences linking survival analysis’ theoretical strengths， handling censored data, hazard modeling, or time-varying risk factors—to the nature of dialogue breakdowns in LLMs.

2. ***Problem Formulation***
- The definition of failure may be too coarse, as treating the first error as collapse ignores possible self-correction. Introducing a tolerance window or semantic distortion metric could yield a more continuous and realistic measure of failure.

3. ***Validation set:*** In the Experiment Setup section, the dataset is split into 80% for training and 20% for testing, but there is no validation set included.
A validation set is typically necessary for hyperparameter tuning and early stopping, so the authors should clarify whether they used cross-validation, a held-out subset of the training data, or another validation strategy to avoid potential overfitting.

4. The used benchmark in this paper only contains 700 questions, which is relatively small and thus lacks sufficient persuasive power to support the generality of the conclusions.

### Questions
1. ***Typos or inconsistency.*** The paper’s readability is poor, mainly due to weak continuity and unclear writing between sections. The transitions are abrupt, and ideas are not logically connected, making it difficult for readers to follow the narrative flow.

- Line 136 Page 3, "time-to-failure" should be "time-to-event". 
- What exactly does the term **“feature”** refer to in this context in Table 1?
- The paper claims to analyze **9 state-of-the-art LLMs**, but based on Tables 1–2, it appears that **9 survival models were applied to a single LLM** instead. Could the authors clarify whether the study involved multiple LLMs or multiple survival modeling approaches on one model?

### Soundness
2

### Presentation
1

### Contribution
2

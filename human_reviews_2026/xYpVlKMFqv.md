# NextQuill: Causal Preference Modeling for Enhancing LLM Personalization

- Decision: Accept (Poster)
- Scores: 4, 4, 6, 6

## Abstract
Personalizing large language models (LLMs) is increasingly important as they are progressively integrated into real-world applications to support users’ daily lives. However, existing approaches often fail to distinguish which components of response predictions by model and ground-truth response in training data truly reflect user preferences, resulting in shallow personalization alignment. In this paper, we introduce NextQuill, a novel LLM personalization alignment framework grounded in causal preference modeling. We approach personalization from a causal perspective, recognizing that model-predicted responses (model side) and user-written ground-truth responses (data side) are both outcomes shape by user history (characteristics) and other context factors. To better capture user preferences, we define causal preference effects as the causal effect of the  user history/characteristics on outcomes from the model/data side. Building on this foundation, NextQuill introduces two complementary alignment strategies: (1) aligning model-side causal preference effects (on predictions) with those of ground-truth data, rather than indiscriminately aligning all predictions, and (2) emphasizing learning the preference-driven ground-truth tokens, identified via data-side causal preference effects, rather than treating all tokens equally. As such, NextQuill shifts the alignment process toward learning from causal preference effects, facilitating more effective and personalized LLM adaptation. Experiments on multiple personalization benchmarks demonstrate that NextQuill substantially improves personalization quality. Code is available at \url{https://github.com/juntaoyou/NextQuill}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper presents NextQuill, a new fine-tuning framework for LLM personalization. It argues that current methods are too “shallow” because they don’t separate true preference signals from generic contextual noise in training data. It takes a causal approach, distinguishing preference-driven from context-driven components in both model outputs and ground truth. It introduces two alignment strategies: (1) matching the model’s causal preference effects to real data, and (2) applying preference-weighted supervision that focuses on preference-driven tokens. The authors report improvements across several personalization benchmarks.

### Strengths
(1) **Clear problem design**: The paper clearly outlines the limitations of existing methods that treat all tokens equally and makes a compelling case for adopting a preference-aware learning approach.

(2) **Novelty**: The work introduces a novel approach to disentangle user preferences from contextual factors. Its dual-sided causal graphs, covering both the model and data sides, provide a clear and elegant theoretical foundation.

(3) **Strong empirical results**: Improvements are consistent across metrics (ROUGE, METEOR, BLEU, BERTScore) and scale with model size. It includes diverse baselines spanning retrieval-based and fine-tuning methods, and ablation studies highlight the importance of each framework component.

(4) **Interpretability**: The word-level preference analysis and case study offer qualitative insights that effectively support the paper’s contributions.

### Weaknesses
(1) **Loss inconsistency**: In the main text, $L_p$ in Eq. 7 aligns MCE to the ground truth label token $y_t$ via cross-entropy. However, in Algorithm 1 in the Appendix, $L_p$ aligns MCE directly to DCE, that is, effect-to-effect rather than effect-to-label. These are not equivalent objectives. If the intended goal is “align model-side and data-side effects,” the algorithmic version matches the claim most literally, but the equation in text uses the ground label token as the target. This critical inconsistency makes the paper's core method ambiguous and, as written, unreproducible for now.

(2) **Possible information leakage in DCE estimation**: DCE is estimated using the model  $f_{\theta_D}$,  which has been trained on the dataset D. This setup raises the possibility of information leakage, as $f_{\theta_D}$ may inadvertently encode knowledge of the data distribution. Consequently, the resulting DCE scores might not represent purely causal estimates but could instead be partially influenced by model memorization.

(3) **Limited Scope of Personalization**: As noted in the limitations, the approach focuses on single-session, token-level personalization without considering long-term user dynamics or cross-session behaviors, which limits applicability to real-world scenarios.

### Questions
(1) **$L_p$ clarification**: A precise and consistent description of the implemented $L_p$ formulation would strengthen the paper. Clearly delineating which version, effect-to-label or effect-to-effect, is used in experiments, and providing a detailed breakdown of the loss components, would help in resolving the ambiguity surrounding this objective.

(2) **DCE estimation methodology**: The current setup of DCE estimation could potentially lead to information leakage. An analysis or justification of why this doesn't introduce bias would be helpful. In case of potential leakage, considering or comparing alternative estimation strategies might be helpful.

(3) **Low-user interaction performance**: While the paper acknowledges that limited user history constrains personalization performance, it might be valuable to quantify this effect empirically. An analysis showing how model performance varies with different levels of user interaction (e.g., short vs. long histories) would clarify the method’s robustness in low-interaction or cold-start settings.

(4) Appendix G states that the method’s validity relies on the assumption of no unobserved confounders. A discussion of potential confounders in this context and how their presence might affect the proposed method would add important context to the paper’s causal claims.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces NextQuill, a LLM personalization framework that leverages causal graph to identify the more informative tokens in the user history. It estimates preference-driven components on the model side by intervening on user history and measuring resulting prediction changes, and on the data side by scoring tokens according to how strongly they reflect user-specific characteristics. NextQuill finetunes the inference LM on the weighted next token loss to align with these identified tokens. The experiment demonstrates mostly positive results for the proposed approach.

### Strengths
S1. Generally, I like the overall idea the authors present in the paper; in the user history, there are important information for personalization, and there is less relevant information. The idea is intuitive, and causal models can be naturally applied to the topic.

S2. The disentanglement between the model-side causal graph and user-side causal graph is also innovative and make good technical contribution. 

S3. Although questions remain, the experiments include sufficient state-of-the-art baselines and commonly used datasets for a fair comparison.

### Weaknesses
W1. Scope of the paper. 
LM Personalization can mainly be categorized into two categories: Personalized text generation [1,2] and personalized preference alignment [3], which are different in terms of the tasks. The paper claims to solve both as L124 claims “…representing text written or liked by the user.” However, the evaluation and experiments do not support the claim on the preference alignment as the author includes no preference optimization baselines [3]. Instead, it focuses on the text generation tasks such as review writings as seen in prior works on personalized text generation [1, 2]. We should not mix these concepts and it can create potential confusion to the readers. Please include more discussion in the related work and clearly call out the scope of the paper. 

W2. The Identified Components. 
The central claim of the paper is that causal graphs can help identify crucial language component for the task of personalization. However, in the experiments, we do not see the components that are identified to be idiosyncratic for user preference. Although Table 2 in the ablation studies demonstrate the effectiveness of including the causal loss in the model training, it does not exhibit the soundness of the causal graph as the improvement can simply come from better SFT alignment.  

W3. LLM-as-a-Judge. 
LLM-as-a-judge has recently been used in personalization evaluation, as seen in [4, 5]. In D.4, we appreciate the author for including the results for LLM-as-a-Judge. What is the scale for the evaluation? The delta seems incredibly large comparing to the lexical metrics such as BLEU and ROUGE. Some analysis and prompt examples would be appreciated. Additionally, LLM-TRSR seems to outperform the other models in LLM-as-a-Judge metric, but is one of the worst performers (0.0465) in traditional lexical metrics (ROUGE-1). I think we need more exploration and analysis into the results. Maybe even including more advanced peer-reviewed LLM-as-a-Judge that is designed for personalization [5].

### Questions
1. Please revisit the scope of the paper and include proper introduction for personalized preference optimization and personalized text generation. Consider refine the problem definition in L124 and carefully improve the definition. 

2. Include the analysis on the proposed causal graph and justify its soundness through experiments. In addition to the ablation study results in Table 2, include the example of identified tokens and their corresponding influence in the SFT. Since we have two causal models, we should compare the effects of including the individual causal component in the experiment. 

3. More analysis on the LLM-as-a-judge results. What is the experiment setting? Prompt of the LLM-as-a-judge? 

4. Compare the performance of LLM-as-a-judge results and traditional lexical results. Especially for LLM-TRSR and the proposed models. What makes the LLM-as-a-Judge so different? 

5. Experiment with LLM-as-a-judge for personalization? [5]

6. Please discuss and include more relevant works, e.g., [1, 2, 3, 6]

In general, I like the idea and would increase my score conditioned on the addressing of the above mentioned points. 

[1] LaMP: When Large Language Models Meet Personalization. Alireza Salemi, et al. 
[2] LongLaMP: A Benchmark for Personalized Long-form Text Generation. Kumar, et al. 
[3] Personalized Language Modeling from Personalized Human Feedback. Xinyu Li, et al. 
[4] Reasoning-Enhanced Self-Training for Long-Form Personalized Text Generation. Salemi, et al. 
[5] ExPerT: Effective and Explainable Evaluation of Personalized Long-Form Text Geneartion. Salemi, et al.
[6] A Personalized Conversational Benchmark: Towards Simulating Personalized Conversations. Li, et al.

### Soundness
2

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
This work presents a framework for personalization, positing that alignment should target causal preference effects rather than uniformly fitting all tokens in a sequence. The method defines two key concepts: 1) a Model-side Causal Effect (MCE), which quantifies the change in next-token predictions when user history ($H$) is included versus omitted; and 2) a Data-side Causal Effect (DCE), which uses a separate model ($f_{\theta_D}$) exposed to the dataset $D$ to estimate which ground-truth tokens were likely driven by user preferences. The training objective combines two components: a preference-weighted cross-entropy loss (which up-weights tokens identified as DCE-positive) and a causal preference alignment loss. This second loss explicitly encourages the model's MCE to align with the ground-truth tokens identified by the DCE. Empirically, the proposed method reports consistent performance gains over retrieval-based and PEFT baselines across two datasets. This improvement is achieved with additional training-time overhead but adds no extra cost at inference.

### Strengths
The paper provides a clear motivation and an intuitive conceptual framing. It proposes a causal framework that moves beyond the naive approach of uniformly aligning all tokens, instead targeting preference-driven components. This causal perspective for personalization and its instantiation are valuable contributions.

The empirical evaluation is thorough, reporting consistent and significant performance gains over strong baselines across several datasets. The results also compellingly show that these improvements scale favorably with increasing model size.

The ablation studies effectively demonstrate the utility of core components (the preference-weighted token loss and the causal alignment loss), indicating that they provide complementary contributions to the final performance.

### Weaknesses
This method can fail when the main preference signal lives in the current query/context $x$. Because it defines “preference-driven” via the with-history vs. no-history contrast and up-weights only $H$-mediated tokens, it systematically down-weights $x$-mediated cues (e.g., “be brief,” tone, formatting). So in cases where $x$ actually carries strong preference evidence, the method can misprioritize or even suppress those signals—i.e., my concern is that it doesn’t work properly when on-the-fly preferences are expressed in $x$.

DCE hinges on f_{\theta_D}—a model “that has seen D”—to identify which tokens are “preference-driven.” However, the assumption that merely having seen D suffices to use f_{\theta_D} as a reliable approximator is not fully justified. Given that the manuscript does not specify what f_{\theta_D} actually is (backbone, training objective, whether it is causally trained, and its personalization performance), I have reservations about the fidelity of Eq. (5). If f_{\theta_D} is causally trained, the paper should describe the objective, explain how it differs from NextQuill, and include f_{\theta_D} as a baseline. If f_{\theta_D} is not causally trained yet already strong at personalization, that weakens the claim that causal effects are necessary and calls for clarification of NextQuill’s added value beyond simply using f_{\theta_D} for personalization. If f_{\theta_D} is not strong, using it to approximate DCE is unreliable. To resolve these concerns, the manuscript should precisely specify f_{\theta_D}, report its personalization metrics, and provide stronger validation that f_{\theta_D} can serve as a reliable approximator for DCE (e.g., small human checks). As written, the support is currently insufficient.

### Questions
Please see weaknesses.

Typo: “personalizaiton” (L134)

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces NextQuill, a causal preference–modeling framework for LLM personalization. The key idea is to focus training on preference-driven tokens in both the targets and the model’s predictions. Concretely, the authors compute model-side and data-side causal preference effects and fine-tune the model with (i) a preference-weighted objective and (ii) an effect-alignment loss. Experiments on four personalization tasks show consistent improvements over retrieval-based and PEFT baselines, supported by ablations.

### Strengths
- **Clear, principled formulation.** Personalization is framed as token-level causal effects, which naturally motivates a two-loss fine-tuning scheme (preference-weighted next-token loss + effect-alignment loss). The causal lens is intuitive and likely useful beyond the evaluated tasks.

- **Effective performance with solid analysis.** The method achieves strong gains over retrieval and PEFT baselines across Amazon review domains and a long-form topic-writing task. Ablations indicate both components (token weighting and effect alignment) contribute meaningfully.

### Weaknesses
- **Limited validation of the causal preference effects.**
Qualitative illustrations are helpful, but deeper evidence is needed that tokens flagged as “preference-driven” correspond to user preferences rather than context artifacts.
Suggestion: Add systematic analyses—e.g., human-annotated token studies; correlation between model-side and data-side effects; negative-control tests—to show the method truly captures preference vs context.

- **Gap between formulation and real-world.**
User histories often entangle preference with contextual factors (topic, recency, item attributes).
Suggestion: Report robustness under noisy or partially mismatched histories; perturb or mask contextual attributes to quantify spillover from context to preference labels.

- **Intervention vs conditioning (identifiability).**
The method effectively treats do(𝐻=ℎ) as equivalent to conditioning on H=h. In real data, unobserved confounders (item popularity, seasonality) can bias effect estimates.
Suggestion: Provide a clearer justification and a sensitivity/backdoor analysis (e.g., synthetic confounders, stratification by proxies) to support the causal claims.

### Questions
**Multiple or conflicting preferences.** Can the framework handle composite or conflicting user preferences (e.g., time-varying or multi-persona) and decompose effects by type?

**Context in histories.** How do you separate preference from context when histories include both? Any diagnostics showing that preference-driven tokens are not merely contextual artifacts?

I

### Soundness
3

### Presentation
2

### Contribution
3

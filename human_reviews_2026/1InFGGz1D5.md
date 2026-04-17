# Capability-Based Scaling Trends for LLM-Based Red-Teaming

- Decision: Accept (Poster)
- Scores: 4, 8, 4, 4

## Abstract
As large language models grow in capability and agency, identifying vulnerabilities through red-teaming becomes vital for safe deployment. However, traditional prompt-engineering approaches may prove ineffective once red-teaming turns into a \emph{weak-to-strong} problem, where target models surpass red-teamers in capabilities. To study this shift, we frame red-teaming through the lens of the \emph{capability gap} between attacker and target. We evaluate more than 600 attacker-target pairs using LLM-based jailbreak attacks that mimic human red-teamers across diverse families, sizes, and capability levels.  Three strong trends emerge: (i) more capable models are better attackers, (ii) attack success drops sharply once the target’s capability exceeds the attacker's, and (iii) attack success rates correlate with high performance on social science splits of the MMLU-Pro benchmark. From these observations, we derive a \emph{jailbreaking scaling curve} that predicts attack success for a fixed target based on attacker-target capability gap. These findings suggest that fixed-capability attackers (e.g., humans) may become ineffective against future models, increasingly capable open-source models amplify risks for existing systems, and model providers must accurately measure and control models' persuasive and manipulative abilities to limit their effectiveness as attackers.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates how the capability gap between attacker and target language models affects automated jailbreaking success. The central contribution is the discovery of a capability-based scaling law: jailbreaking success (ASR) scales linearly with the capability gap in logit space, with high predictive power across most tested models. The authors demonstrate that this relationship holds across different attack methods (PAIR and Crescendo), though stronger attacks shift the difficulty curve without changing the fundamental slope. The paper also reveals that social science abilities correlate more strongly with jailbreaking effectiveness than STEM capabilities. Additionally, the paper provides methodological insights showing that judge capability affects only prompt selection rather than generation quality, validating cost-saving approaches for red-teaming. Using the discovered scaling law, the authors forecast that human red-teaming will lose effectiveness once models surpass human-level capability.

### Strengths
- **Comprehensive empirical evaluation at scale.** The paper tests 22 attacker models against 27 target models across 50 harmful behaviors using two attack methods.

- **Clear quantitative scaling relationship.** The central finding that capability gap predicts attack success with high correlation provides a simple, predictive framework that works well for the majority of tested models.

- **Novel insight about psychological capabilities.** Section 6.1's finding that social science abilities (psychology, philosophy) correlate more strongly with jailbreaking success than STEM capabilities is novel.

- **Transparent treatment of outliers.** Rather than obscuring models that don't fit, the paper explicitly identifies outliers (Llama2 family, Llama3-8B) and acknowledges that heavy safety tuning can decouple defensive capability from general capability.

- **Attack method comparison reveals fundamental invariants.** Section 6.3 shows that different attack methods shift the difficulty curve but preserve the underlying slope, suggesting the scaling relationship reflects fundamental attacker-target dynamics rather than method-specific artifacts.

### Weaknesses
The paper acknowledges that 'heavy safety tuning can extend a system's lifespan against stronger attackers', but the framework doesn't incorporate this insight systematically. The scaling law works well for models with no or weak safety alignment, but fails for models with 'exceptional' alignment (Llama2 family, Llama3-8B). This suggests safety alignment strength is an important independent variable that deserves explicit treatment rather than being absorbed into per-family baselines or outlier exclusion. The paper also provides no per-target or per-family scaling curves for frontier models (Claude, GPT, Gemini), which makes it impossible to assess whether heavily-aligned frontier models follow the observed pattern or represent 'exceptional' cases that the framework cannot capture. For forecasting purposes, the key question becomes: will future models follow the observed pattern (Qwen, Mistral) or the 'exceptional' pattern (Llama2)? The paper's forecast implicitly assumes the former but cannot validate this assumption on current frontier models—the very models whose alignment practices are most likely to inform future frontier development.

### Questions
Figure 3 (right) appears to show Qwen2.5 models with similar vulnerability across different MMLU-Pro scores, while Figure 4 (bottom left) demonstrates a clear capability-based scaling law for the same family. Can the authors clarify this apparent discrepancy?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
Summary  
The paper tries to create a Scaling law for Jailbreaking of models. It's setup is based off of an harmfully fine-tuned attacker and judge, where the attacker jailbreaks a defender and the judge evaluates success or failure.  They rank the models capabilities based off of their MMLU-Pro scores and then analyze different pairings along with capability vs jailbreak attempts at N attempts and many other combinations.

### Strengths
Pros  
- Semi-comprehensive testing to try to figure out if there's a scaling law for jailbreaking
- Useful for forecasting

### Weaknesses
Cons  
- It doesn't seem to highlight limitation that "jailbreaks" might evolve or change forms (imo) namely we might care about different jailbreaks or different sensitive topics in the future.
- I couldn't find if there was a manual QA somewhere. The setup is LLM judges if LLM attacker can jailbreak defender but I'm unsure of false positives or false negatives in this setup.
- The "what makes a good attacker" might be heavily mis-representing the "truth" as models scale they might tend to do better asymetrically on "easier" subjects or questions (like psychology).  This might just be another scaling law plot in disguise. I'd like to see the variance of improved overall performance or the increased compute time variance removed to actually be able to say that humanities capabilities matter more than STEM.

### Questions
Notes  
- Figure 4
It seems as though the regression was messed up or inaccurately plotted on the llama family data

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
4

### Summary
This paper investigates automatic red-teaming, in particular what happens if you vary the strength of the attacked and attacker models. The findings show that the strength of the attacker plays a central role in being able to successfully red-team a model. The paper analyzes this from a perspective of MMLU performance of the two models and identifies that the attack success probability follows mildly predictable trends.

### Strengths
- I really like the research direction and the work presented in the paper is extensive and comprehensive
- The experiments and approach are innovative with interesting insights which spark many potential ideas for future followups

### Weaknesses
- **Scaling Law Misdirection**: My main point of critique is that the paper trying to turn things into a scaling law that really shouldn’t be one. In a way, the notion of there being “scaling laws” is detracting from a lot of the valuable insights this paper is providing. Many curves seem very forced (e.g., in Fig 4) and just drawn through a cloud. I am pretty sure even a linear fit would achieve a similar R^2. There are many parts where correlations are presented as causations, and as discussed in the next point, the experiment rely on a few critical assumptions that may not generalize. As such, I would strongly recommend toning down the "scaling law" claims in this paper. 

- **Critical Assumptions**: The paper explores relationships of attacks between one benchmark (MMLU-Pro) and one notion of harmfulness (HarmBench). As such, any claims need to be framed within those terms. By no means are the results generalizable scaling laws. Moreover, since the Llama 2 system prompt is used throughout many of the experiment, it further biases the results since not all models were optimized for this prompt. 

- **Explanations are missing**: There are many experiments explaining what is happening, but not many that try to uncover underlying mechanisms at hand. As a result, the results are not immediately actionable other than taking away the advice to "use the best possible model as attacker" which seems rather obvious. 

- **Alternative result interpretation**: It could be the case that models with stronger MMLU results also had more safety tuning since they tend to have larger teams behind them. As a result, the key result could just measure the correlation between the post-training effort put into safety and NLU benchmarks. 

- **6.1 seems contrived**. While there are correlations between the various MMLU sub-categories, there are a myriad of alternative explanations (e.g., that large models are overfit to STEM categories because of a focus on IMO-style problems). The sections thus also conflates correlation and causation. Moreover, the section has a massive overclaim. Good results in multiple choice tests should absolutely not be categorized as “might rely on psychological insights and persuasiveness” as the authors suggest. 

Minor:
- Don’t use white-box vs black-box. These terms have racist connotations. 
- The related work section is extremely disappointing. Research existed for longer than 2024.

### Questions
n/a

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents a large-scale empirical study on the dynamics of LLM-based red-teaming, reframing the problem from one of absolute model capabilities to one of a relative "capability gap" between the attacker and the target. By evaluating over 600 attacker-target pairs with two distinct LLM-based jailbreak methods (PAIR and Crescendo), the authors establish a "jailbreaking scaling law." This law demonstrates that ASR is not linear but follows a predictable sigmoid function of the capability gap, which they measure using MMLU-Pro benchmark scores. Key findings include that more capable models are both better attackers and more robust targets, and that success in red-teaming correlates more strongly with social science and persuasion-related capabilities than with STEM knowledge. Ultimately, the work forecasts that as AI models become more capable, the effectiveness of any fixed-capability attacker, including human red-teamers, will inevitably and predictably decline.

### Strengths
- The analysis is built upon "more than 600 attacker-target combinations", providing a robust empirical basis for its conclusions. The heatmap in Figure 2 visualizes this extensive dataset, lending significant weight to the observed trends and the derived scaling law.

- By fine-tuning models to remove their safety guardrails, they "eliminate the attacker's refusal as a confounding factor". This allows the study to focus on a model's raw ability to craft adversarial prompts, rather than its willingness to engage in the red-teaming task, which is a critical distinction for a clean analysis.

- The "Capability-Based Jailbreaking Scaling Law" (Figure 4) models ASR as a sigmoid function of the capability gap. This provides a concrete, predictive tool for forecasting the future efficacy of red-teaming efforts and reasoning about the security of deployed systems, which is a significant contribution over merely reporting ASR metrics.

### Weaknesses
- The paper's central narrative is built on the foundational assumption that general academic benchmark performance (MMLU-Pro) is a valid proxy for the highly specific skills of both offensive jailbreaking and robust defense. This assumption is questionable and introduces a potential circularity. The authors first use MMLU-Pro to define the "capability gap" axis and then, in Section 6.1, show that ASR correlates strongly with the social-science splits of MMLU-Pro (Figure 6). While this correlation is an interesting finding, using MMLU-Pro as the primary independent variable from the outset presumes it is the correct measure. The skills for jailbreaking, such as creative deception, exploiting logical loopholes, and social engineering, are not directly measured by multiple-choice questions. The paper's own data points to this flaw: it dismisses the "early Llama models (Llama2 and Llama3-8b)" as "outliers" (Figure 3) because their robustness is higher than their MMLU-Pro score would predict. These are not mere outliers; they are significant counter-examples suggesting that MMLU-Pro is an incomplete, and at times incorrect, proxy for defensive capability, and that specific safety alignment procedures can break the proposed scaling relationship.

- The paper presents the "unlocking" procedure as a clean removal of safety alignment to reveal a model's intrinsic attacking ability. However, the methodology of fine-tuning on harmful datasets like "BadLlama" and "Shadow Alignment" does not simply revert a model to a neutral base state; it actively trains it to become proficient at generating harmful content in a particular style. This introduces a critical confounder: the experiment may be selecting for models that are better at adapting to this specific fine-tuning task rather than measuring a general, latent "attacking capability." This is evidenced by the authors' own observation of an "unwanted unlocking artifact" where models "overfit to harmful content in the red-teaming prompt". While mitigated with instruction-following data, this core issue remains. The "unlocked models" are fundamentally different from the original ones, meaning the capability scores on the x-axis of Figures 3 and 4 belong to a different model than the one whose attacking performance is being plotted, undermining the directness of the claimed relationship.

### Questions
None

### Soundness
2

### Presentation
2

### Contribution
2

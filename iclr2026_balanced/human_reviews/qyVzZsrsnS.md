## Human Reviewer 1

### Summary
The paper introduces a model diffing framework, the Activation Difference Lens (ADL), designed to extract clearly readable traces of LLMs' internalized objectives resulting from narrow finetuning. The core methodology involves measuring the differences between the activations (δ) of a base model and a narrowly finetuned model on the first few tokens of random pretraining corpus data. Their methodology is clearly explained, leveraging two key interpretability tools applied to the activation differences:  Logit Lens, Patchscope (modified), which reveals relevant tokens), and steering (which amplifies the model’s tendency to generate text highly similar to the finetuning data). The paper develops an LLM-based interpretability agent which uses these ADL results and LLM graders to form and verify hypotheses about the finetuning objective.

### Strengths
The main contribution is the demonstration that these few-token activation differences encode salient traces of the training objective. The empirical study, spanning a wide range of model organisms and architectures, proves the ADL framework is highly effective compared to existing blackbox baselines using only simple prompting. Through causal ablation analysis and loss calculation, the paper makes a sound argument that the easily detectable biases highlighted by ADL are somewhat tied to overfitting to the semantically homogeneous finetuning data. Finally, through mitigation experiments, the authors proposed an effective method to avoid these strong biases by mixing unrelated pretraining data into the narrow finetuning datasets, which substantially reduces the detectable bias. The paper also highlights a warning to the interpretability and AI safety communities. These findings suggest finetuning signals may be artificially strong and overpower traces from standard broader finetuning. Therefore, using such model organisms as proxies for realistic scenarios may compromise their validity.

### Weaknesses
In general, this study brings important insights to narrow finetune and model organism study. However, some caveats and additional experiments could make even sounder arguments. 

- The empirical study involves extensive usage of LLM grader. Although it is mentioned as a limitation, It would be nice to see some expansion on generalization of such a framework with other LLM graders other than gpt-5-mini.
- The causal effect analysis is only done on three models, relatively small sized. Expanding the study to a wider range of models architecture and of model sizes will help make a stronger argument of overfitting. Further, the inconsistent result from Gemma makes the conclusion harder to believe. Experiments on the older Gemma model could help to further ground the hypothesis for its inconsistent behavior.

### Questions
- In section 4.2, it is discussed that similar activation difference analysis was also done between the base model and the narrowly finetuned model, where bias detectable similar to that of chat model and finetuned model. Although the argument is sound, it seems the setup overlooks the potential impact of activation difference between base model and chat model. Can a similar analysis be done between the base model and the chat model? Would that help control for confounders?

### Soundness
3

### Presentation
3

### Contribution
3

### Rating
8

### Confidence
4

---

## Human Reviewer 2

### Summary
This paper finds that narrow finetuning of LLMs (a popular technique to study model behavior) creates detectable biases in activation differences between base and finetuned models, which can be used to identify the finetuning objective. The authors apply model diffing techniques (Patchscope, Logit Lens, steering) to analyze a broad range of "model organisms" (narrowly finetuned LLMs) and demonstrate that: (1) activation differences on early tokens of unrelated text encode finetuning domain information, (2) an interpretability agent with access to these diffing techniques that can identify finetuning objectives better than an agent with only black box access, and (3) the narrow / monosemantic nature of the data leads to these strong biases, so mixing pretraining data during finetuning mitigates biases through diversification, albeit also weakening the desired effects of narrow finetuning.

### Strengths
1. The paper finds a flaw in current AI safety practice (where narrow finetuning is used extensively), so the findings have practical significance in AI safety research (although the authors do not provide a compelling alternative/solution, as discussed below).
2. The experimental scope is quite comprehensive, covering 33 organisms across 4 families and 7 models from 1B up to 32B parameter scales.
3. The interpretability agent experiment provides an objective automated assessment of how useful / exploitable these activation differences are. These experiments also should be reproducible as the results are reportedly stable across runs.
4. The authors go beyond pure observation of the phenomenon by conducting (well-designed) causal ablation experiments to trace biases back to the semantic homogeneity of the data.

### Weaknesses
1. The scope of the paper is quite narrow (no pun intended) because narrow finetuning itself is primarily used as a safety / interpretability tool while in practice training data tends to be more diversified. The authors contend that in more realistic settings (such as domain-adapted vision-language models) the phenomenon is observed to a lesser extent.
2. The paper relies on an interpretability agent based on gpt-5-mini to show that the studied activation biases can be detected easily with the right tools (ADL) while blackbox access to the model is insufficient. The effectiveness of both agents may be highly sensitive to prompt design and it is not clear that the authors tuned both agents equally well.
3. In general, the paper relies heavily on LLM graders. The credibility of the claims would be strengthened by including some human judgments to spot-check whether the graders are calibrated well.
4. The proposed mitigation strategy (mixing in pretraining data) creates a trade-off between strength of activation biases and desired effects of narrow finetuning, and the authors provide no clear guidance on how to best navigate this trade-off.

### Questions
1.  What is the reason for using activations from the middle layer? This decision seems quite ad-hoc given that activations vary a lot across layers. Can you observe activation biases in lower or higher layers too?

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The work uncovers biases in narrowly fine-tuning models' activation spaces and hypothesizes the origins of said biases, and provides a possible direction to mitigate such biases.

### Strengths
1) I think the paper is written very well. 
2) I think the goal and contribution of work are very relevant to interpretability research as a whole. 
3) The experiments are sound and aid the central claims of the work. 
4) The finding is very interesting and well-grounded in current literature. Although the paper could use some more explanations of the methods utilized in the work, such as PathScope. I encourage the authors to add further explanations about this methodology in the final iterations of the work.
5) Mixing pre-training data with the fine-tuning one is an interesting approach; I would like some more emphasis on this in the final iteration of the paper. 
6) Overall, the work aims to present and discuss key facts about fine-tuning for narrow use cases, particularly the data dependence of bias in fine-tuned models.

### Weaknesses
The major weakness of note is that the bias mitigation method proposed lacks a study of the impact of injecting pre-training data into the fine-tuning object. Both in the sense of measuring the domain-specific capabilities impacted by the model, and the added cost of adding more data to the fine-tuning objective. Although the work does claim that adding pre-training data reduces fact alignment scores, a more rigorous analysis would be appreciated. 

Another weakness, more so of a comment, to highlight would be that the methodology presented is just a collection of prior known/used methods, although this doesn't undermine the novelty of the findings; highlighting the derivative nature of the methodology would help elucidate the contributions of the work. 

Even though, as mentioned previously, i think the paper is written very well, continuous references to the appendix break the flow of the paper. I would encourage that in the final iteration of the work, to add these appendix section to the main paper or leave some as a footnote.

### Questions
Would the results be consistent across multiple graders, it would be nice to know the amount of noise that could be mitigated if we use multiple graders.

### Soundness
3

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3

---

## Human Reviewer 4

### Summary
The paper introduces ADL method which uses model diffing techniques to investigate traces left by narrow finetuning. The traces are visible even for unrelated inputs and the authors identify, reason and provide strategies to mitigate this effect.

### Strengths
* Thorough experimental setup and extensive results
* Novel methodology that effectively leverages existing techniques to identify traces  
* The analysis provides very important inferences which will be beneficial for the community about safety in narrow fine-tuned models
* It also provides suggestions on how to mitigate this effect

### Weaknesses
The causal effect is positive for Gemma3 which contradicts the hypothesis and warrants more investigation

### Questions
* The authors specify in line 161 that they focus on the middle layer. Could they expand on why?
* Edit suggestion: Explicit mention of the method name (ADL) when introducing the approach in the abstract would be good

### Soundness
4

### Presentation
4

### Contribution
3

### Rating
8

### Confidence
3
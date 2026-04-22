# Are Reasoning LLMs Robust to Interventions on their Chain-of-Thought?

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 6, 4, 6, 2

## Abstract
Reasoning LLMs (RLLMs) generate step-by-step chains of thought (CoTs) before giving an answer, which improves performance on complex tasks and makes reasoning transparent. But how robust are these reasoning traces to disruptions that occur within them? To address this question, we introduce a controlled evaluation framework that perturbs a model’s own CoT at fixed timesteps. We design seven interventions (benign, neutral, and adversarial) and apply them to multiple open-weight RLLMs across MATH, SCIENCE, and LOGIC tasks. Our results show that RLLMs are generally robust, reliably recovering from diverse perturbations, with robustness improving with model size and degrading when interventions occur early. However, robustness is not style-invariant: paraphrasing suppresses doubt-like expressions and reduces performance, while other interventions trigger doubt and support recovery. Recovery also carries a cost: neutral and adversarial noise can inflate CoT length by more than 200%, whereas paraphrasing shortens traces but harms accuracy. These findings provide new evidence on how RLLMs maintain reasoning integrity, identify doubt as a central recovery mechanism, and highlight trade-offs between robustness and efficiency that future training methods should address.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper studies how robust are reasoning models to disruptions of their own reasoning traces. To study this question, authors introduce a comprehensive intervention framework that contains benign, neutral, and adversarial interventions and analyze models' behaviors including 1) performance on different data domain, different intervening timesteps, different models, 2) doubtfulness rate, and 3) CoT length. The results indicate that reasoning models are robust in general, but the robustness is not style-invariant.

### Strengths
1. The paper is well-written, easy to follow, and the findings are clear. The findings by the authors could give a big contribution to the community.
2. The motivation of this work (a new benchmark for reasoning robustness) is valid and sound. The early parts of Section 3 well describes this point.
3. Authors comprehensively analyze the observed phenomenon in Section 4. This helps readers to understand the limitation of the robustness of the reasoning models more deeply. -- The finding in line 416-418 is very interesting.

### Weaknesses
1. In Section 3 line 135, "RLLMs outperform non-reasoning models of comparable and even larger size, ...". In order to support this sentence, I think it is more fair and reliable to compare reasoning models and their non-reasoning counterparts (e.g., Qwen3 think vs no-think, DeepSeek V3 vs DeepSeek R1). Could you report results for one of these?  
2. In line 187, "To validate semantic preservation, we manually ..." So what is the result of this manual comparison? Since the paraphrasing reasoning leads a key finding, ensuring that the paraphrasing is performed well by the models is very important to support authors' claims.  
3. Why is "Unrelated CoT" adversarial? isn't it neutral? -- inserting "unrelated" factual content is neutral now. Other two types of adversarial reasoning contain incorrect information, whereas this type does not. By definition by authors, neutral interventions mean "irrelevant information into the CoT", which also fits to this type as well. I understand authors' intention, so to improve soundness of this work, clarification is needed.  
4. The findings in Section 4.2 The role of doubt in reasoning is not surprising.  
5. In Table 5, Llama-nemotron-8B shows the highest increase of CoT length in paraphrasing reasoning scenario. Why does this happen? This result could indicate that the effect of paraphrasing is only applied to R1-distill-qwen/llama models. Please justify this result.

### Questions
1. What if the intervened reasoning is given to another model that is different from the model that originally generates the reasoning trace?  
2. Is there any reason that authors try to fix the timestep instead of reasoning step? If authors try this, what's the reason not to choose this strategy?

### Soundness
2

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
This paper investigates the robustness of Reasoning Large Language Models to interventions within their own generated CoT reasoning processes. The authors introduce a controlled evaluation framework where they interrupt a model's correct CoT at various timesteps and inject one of seven different perturbations. These interventions are categorized as benign (e.g., paraphrasing), neutral (e.g., inserting random text), and adversarial (e.g., adding a wrong reasoning step). The same model is then prompted to continue its reasoning, allowing for a direct measurement of its recovery capabilities. The study evaluates multiple RLLMs across MATH, SCIENCE, and LOGIC domains. Key findings indicate that RLLMs are generally robust to such disruptions, with robustness increasing with model scale. However, this recovery comes at a computational cost, often significantly inflating the CoT length. Finally, the paper identifies the expression of "doubt" as a key recovery mechanism and reveals that models are not style-invariant, as paraphrasing can suppress doubt and degrade accuracy despite preserving semantics.

### Strengths
1.  By intervening directly in a model's own reasoning trace, the authors create a clean and realistic testbed for self-correction, which is a significant step forward in robustness evaluation.

2.  The study is impressively broad, covering 9 models, 3 domains, 7 intervention types, and 5 timesteps. This thoroughness lends high credibility to the conclusions and suggests that the findings are generalizable.

3.  The paper is well-written. The research question, methods, and results are communicated with great clarity, making the experiments easy to understand.

### Weaknesses
1.  While identifying "doubt" is a major strength, the analysis relies on an LLM-based classifier to label sentences. This approach, while pragmatic, is somewhat superficial. The paper would be significantly strengthened by a deeper investigation into how "doubt" is represented internally. The activation analysis in the appendix is a good first step, but it feels underdeveloped. A more detailed analysis connecting specific internal states to the expression of doubt would benifit the contribution of the work.

2. The interventions are all single-step disruptions. In many real-world scenarios (e.g., interacting with a faulty tool), errors can be more subtle or cascading. For instance, a model might accept a slightly incorrect tool output and then build several subsequent reasoning steps upon it. The current framework does not evaluate robustness to such persistent or multi-step correlated errors, which may be harder for the model to detect and recover from.

3.  The paper excels at diagnosing the problem but is light on proposing solutions. The conclusion suggests that future training methods should address these weaknesses (e.g., style invariance, recovery efficiency), but it offers no preliminary experiments on how this could be achieved. For example, could data augmentation with paraphrased CoTs during finetuning improve style robustness?

### Questions
1.  Regarding the LLM-based "doubt" classifier: How was its performance validated? What is its estimated accuracy, and what are some examples of sentences it correctly or incorrectly classifies as expressing doubt? Could you elaborate more on the findings from the activation analysis?

2.  The interventions are currently single-step. Have you considered experiments with multi-step interventions or cascading errors (e.g., where a wrong fact is introduced and then used logically in the next step)? Do you hypothesize that the "doubt" mechanism would still be effective in such scenarios?

3.  The finding that paraphrasing hurts performance is intresting. It suggests models overfit to a particular reasoning "style." Based on this, what are your thoughts on potential training strategies to mitigate this? Would training on a more stylistically diverse set of CoTs (e.g., via paraphrasing augmentation) be a promising direction?

4.  Could you clarify the process for the "Continuation with other model" intervention? Is the other model a weaker, non-reasoning-specialized model?

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
5

### Summary
The paper studies how robust reasoning models (RMs) are to perturbations in their reasoning trace. They find that accuracy does not suffer as much, but there is an impact on efficiency.

### Strengths
- Efficiency of RMs is an important consideration. While it is not the explicit goal, I think the paper's results on efficiency are interesting and the dataset can help study efficiency further. 
- Reasonable set of interventions and results across models. 
- Ablations on why RMs are able to recover (using "wait" -like tokens) and how the interventions increase inference time per answer. I think this is the strongest part of the paper.

### Weaknesses
- Some of the results need more examination. For example, table 5 shows that benign rewrites lead to drops up to 60%, but then Table 6 shows that there is no drop in CoT length across all intervention timesteps (except 0.9).
- There are interesting observations, but the insight is weak. For example, introducing wrong reasoning increases accuracy robustness and paraphrasing reduces accuracy. Surely, there are some confounding factors here that can be explored?
- Many of the results are "observations", but no solution is motivated or discussed. For example, if certain kinds of interventions lead to higher use of "doubt tokens", then can it be used to improve accuracy (without introducing adversarial interventions)?

### Questions
I think there are some confounding factors at play. 

1. It seems counter-intuitive that introducing benign rewrites reduces accuracy, whereas introducing wrong reasoning improves accuracy. Can the authors unpack this observation? Is it the writing style? What if the benign rewrite is written in a more tentative style or a question? It will be good to do some qualitative study/ablation to see what the cause of robustness is.

2. Same question for efficiency. How can benign rewrites reduce CoT length and accuracy, while other (stronger) interventions don't lead to accuracy drop? Something else may be at play, beyond the type of intervention defined by the authors.

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper investigates the robustness of reasoning process when interruptions are injected in the middle of reasoning trace. 
They introduce unconventional setting where correct pre-generated reasoning traces from RLLMs are produced beforehand, and at each reasoning step in the CoT trace, they inject three types of noises. The benign injection continues the generation from other reasoning model or simply introduce a paraphrasing which do not change the original semantics of the reasoning trace. Neutral injection is adding random characters or documents from wikipedia, whereas adversarial perturbation is to inject unrelated cot trace or incorrect facts. 

They test the reasoning robustness in three domains, math, science and logic. And for five difference RLLMs, they analyze the recovery behavior of each model. Starting from the observation that RL trained reasoning models are more capable of finding the reasoning errors compared to conventional large language models, based on the evaluation on BigBench-mistake, this paper builds a novel benchmark that systematically evaluates self-correction and self-reflection abilities. They define the robustness metric by checking whether the language model is able to produce final correct answer, despite the intervention at the intermediate reasoning steps.

### Strengths
The dataset builds over large number of existing benchmark dataset, extending over various science, math, and logic domains.
Moreover, this paper experimented over wide variety of reasoning models including, R1 distill, exaone, nemotron, phi, and QwQ. 
The discovery that LRMs are generally robust to various types of intervention, regardless of the benign, neural, or adversarial .types.
Also, it is notable that LRM generation length inflates the most when interrupted with random texts for recovery.

### Weaknesses
The largest concern is that the reasoning trace interruption scenario is extremely far from realistic usage cases, which assumes that the model reasoning will be abruptly interrupted by external signals. Moreover the interruptions are hardly meaningful in logical or semantic sense, since it introduces noises that are completely irrelevant with previous context and model generations. Rather than injecting random text from external sources, what if the model is injected with reasoning trace that leads to wrong answer? Does doubt expression contribute to successfully recovering from this adversarial injection?
Also, it would be helpful if the authors could augment their explanation on realistic cases where random interruptions might happen during the model generation.

### Questions
Please answer to the weakness above.

### Soundness
3

### Presentation
4

### Contribution
3

# Output Supervision Can Obfuscate the Chain of Thought

- Avg Score: 7.00
- Decision: Accept (Poster)
- Scores: 6, 8, 6, 8

## Abstract
Recently, OpenAI (2025) showed that training against a chain of thought (CoT) monitor can cause obfuscated CoTs, which contain bad behavior the monitor cannot detect. They proposed to keep CoTs monitorable by training only against output monitors that do not have access to CoT. We show that such training can still cause obfuscated CoTs via two mechanisms. First, when a model is trained to produce a safe-looking output, that model may generalize to making its CoTs look safe. Second, since later tokens are conditioned on earlier ones, safe‑looking CoTs may increase the likelihood of safe outputs, causing safe-looking CoTs to be reinforced. We introduce two mitigations to address these two issues, which achieve a Pareto improvement in terms of monitorability and task performance compared to regular training. To our knowledge, we are the first to identify and mitigate these problems. Our work implies that preserving CoT monitorability is more difficult than previously thought; we suggest practical guidelines for AI developers to maintain monitorable CoTs.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This work discusses an interesting phenomenon; supervising the model outputs in order to prevent specific properties appearing in them, causes a "spillover" effect where the same properties are also removed (i.e., obfuscated) from the chain-of-thought (CoT). The authors attribute this effect to two separate mechanisms, parametric feedback and conditional feedback, obtained via analyzing the structure of the policy gradient on a loss that involves the task and an output obfuscation loss. The first is due to the parameter configurations leading to output obfuscation also generalizing to CoT obfuscation whereas the second is due to the observation that obfuscated CoTs might be needed in order to produce obfuscated outputs. Based on the above observations, the authors propose a specific remedy for each mechanism; they zero-out a specific gradient term in order to prevent conditional feedback and use separate models for the CoT and output in order to prevent the parametric feedback. Finally, they demonstrate on three separate settings that feedback spillover occurs and show how the proposed solutions can help prevent it.

### Strengths
- This work highlights a novel observation, feedback spillover, that manifests when doing output supervision, the suggested way to avoid CoT obfuscation when training against monitors. This is a timely contribution to the emerging field of CoT monitoring.
- The experimental setups to measure CoT obfuscation as well as the presentation of feedback spillover and the methods to tackle it are clear and easy to follow. 
- The insights and recommendations to practitioners are valuable and actionable.

### Weaknesses
- The solutions provided by the authors feel like a "band-aid"; they just push the spillover contributing terms of the loss gradient to zero, which might negatively affect the model performance and overall optimization of the specific loss employed (e.g., what is the impact of the "mind & face" strategy on just the task reward? Is it lower than the standard one where the same model is used for the CoT and answer?). The contribution of this work would have been stronger if the authors instead articulated what loss functions might be better / do not lead to feedback spillover when doing output supervision.
- Even though the authors clearly define parametric and conditional feedback spillover, they do not measure it in the experiments. This work can be improved by showing what kind of spillover exists in each of the experimental setups, thus providing extra motivation / explanation for why, e.g., "reward targeting" or "mind & face" might work better.

### Questions
Overall, I am leaning on the positive side for this work due to the useful insights it provided. Having said that, addressing the weaknesses and comments below will further convince me. 

- How does feedback spillover manifest when the CoT is not necessary for better downstream task performance (i.e., cases where the model does not need to put critical information in it)? Intuitively, if one puts pressure on specific structures for the output this would also force the CoT to change when the model depends strongly on CoT for solving a task.
- How does the spillover effect change during training in the simple case of "Task reward + output monitor" without "reward targeting" and "Mind & Face"? Does it emerge from the get-go or at the end of training? This would highlight whether the spillover could be solved with other means, e.g., regularization or early stopping. 
- What were the architectures for the Mind and Face at each of the experimental settings? Was it the same model at different checkpoints or a completely different model altogether? Did the Mind and Face have similar model capacity? This is important information that needs to be in the main text as it can highlight what practical limitations this strategy has, especially given that it requires having access to two separate models.
- Minor typo: in equation 21 in the appendix there is a missing dependence on $\theta$ on the right hand side.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
- This paper demonstrates that supervision of LLM outputs can “spillover” to increase rates of obfuscated CoT, demonstrating this effect in multiple environments.
- They identify two mechanisms by which this can occur: parametric feedback spillover and conditional feedback spillover, and provide a theoretical analysis of these phenomena.
- This theoretical analysis serves to motivate two corresponding mitigations. 'Reward targeting' mitigates conditional feedback spillover by preventing gradients from the outputs flowing through CoT tokens. 'Mind and face' uses separate models to generate the CoT and output, respectively, thus preventing sharing gradients over the same parameters.
- They show these mitigations can, to some extent, reduce feedback spillover, though results are varied and often task performance is impacted.
- The paper ends with important recommendations to developers, based on their findings

### Strengths
- The demonstration of feedback spillover seems original and significant. CoT monitorability is increasingly highlighted as a promising and important approach for safety. The existence and strength of the feedback spillover effect here is surprising. Naive safety approaches may not have otherwise considered the possibility of this effect. Developers should indeed take note of these results and follow the provided recommendations. I am excited to see follow-up work further investigating this phenomenon and improving corresponding mitigations.
- The paper provides a clear theoretical analysis that explains the potential mechanisms behind the phenomenon. This is a valuable contribution to our understanding of feedback spillover.
- The initial attempts at mitigating feedback spillover are well-motivated, based on the prior theoretical analysis. Whilst the mitigations are not perfect, these are valuable initial attempts and they highlight the difficulty of mitigating this effect.

### Weaknesses
The settings are narrow and somewhat toyish.

There is a lack of focus on the extent to which the models reasoning is “load-bearing” - does the model require its reasoning to optimize task reward in any of these settings? This seems an important distinction to be aware of when interpreting the results. It would be valuable to compare results in environments where reasoning is and isn’t important for the task reward. Relatedly, do the authors think the reward targeting mitigation would significantly hurt task reward in settings where reasoning is required?

While the theoretical analysis of feedback spillover is strong, it could be nice to see more results that provide evidence and a more detailed, empirically-grounded understanding of the two spillover mechanisms. The mitigation results do seem to bear on this, but further depth of analysis could be provided.

There is not many details provided regarding why these particular models and sizes were chosen. Only Qwen-3 models are used, and it is unclear how specific these results are to these models, and to these smaller sizes.

### Questions
- I sometimes had some conceptual confusions when reading the results section. I was often unclear how exactly the experimental setup could map to a realistic safety-related scenario (e.g., in the maths setting, should we think of “expanding the derivative” as a misaligned action that is incentivized by a misspecified reward?). I was initially unclear whether the mitigations should seek to maintain a high task reward on the pareto frontier - the high-task reward corresponds to an “undesired” action - this could be clarified.
- In the maths environment it seems the model learns to perform the math in an entirely different way, rather than obfuscating its initial approach? This seems a notable distinction from “obfuscation” of reasoning?
- I found it confusing when the “Training Reward” metric included the output penalty for “no penalty” training runs which did not use the penalty in their training reward.
- To what extent did you optimize the experimental settings you used to achieve these spillover results? Did you try other environments where no spillover was achieved?
- Could you provide basic details of your RL setup in the main paper? (e.g., what algorithm did you use?)
- Could you provide training curves? It would be interesting to see the initialization points of the various reward terms, and how the rewards evolve over time.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The paper introduces the problem of feedback spillover when training LLMs with output-only supervision, especially modern large models trained with RL. It shows a gap between standard LLM evaluation metrics and genuine reasoning ability, which can limit the interpretability and safety of using CoTs to understand LLMs, undermining the monitorability of CoT.

Additionally, the authors introduce two mitigation methods for feedback spillover, "Mind & Face" Models and Reward Targeting.

### Strengths
- Introduces the core concept of feedback spillover, which can cause Obfuscated CoT,

- Monitoring the logical soundness of the CoT process as opposed to output only during training is quite novel.

- Empirically showed that OS-Models have higher accuracy than CoT-Models, but suffer from invalid CoT monitoring.

### Weaknesses
- The "Mind & Face" model using disjoint parameters to avoid obfuscation does make sense; however, I'm concerned on the computation expense and the possibility of error accumulation, which could harm accuracy or lose some of the supervision signals during training from either the CoT process or the outputs.

### Questions
- Feedback spillover is defined for RL paradigm, but does the same formulation still hold for supervised fine-tuning?

- Does this phenomenon occur more for even larger base models?

- The paper measures spillover through lower CoT monitor scores. To what extent can this reflect safer CoT learning / what are the limitations of this scoring method?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper investigates feedback spillover, a phenomenon where training a model against an output-only monitor (rather than a CoT monitor) still leads to obfuscated chains of thought (CoTs). The authors argue that output supervision can indirectly shape CoTs through two mechanisms:

* Parametric spillover: Parameter updates that make outputs look safe generalize to CoTs.
* Conditional spillover: Safe-looking CoTs increase the likelihood of safe outputs, reinforcing obfuscation.

They propose two mitigations:

1. Mind & Face: Separate models for CoT ("mind") and output ("face").
2. Reward targeting: Modify gradients to prevent reinforcement of CoTs that lead to safe outputs.

Experiments in three RL environments (polynomial derivative factoring, QA with hints, multi-turn terminal interaction) show strong spillover under output penalties and partial success of mitigations. The paper provides a formal gradient decomposition, empirical demonstrations, and practical recommendations for developers.

### Strengths
* **Novel insight**: Challenges the assumption that output-only monitoring preserves CoT monitorability. This is a significant contribution to alignment and oversight research.
* **Clear theoretical framing**: The gradient decomposition and formal conditions for spillover (Eq. 5–7) are well-motivated and connect intuitively to the observed behaviors.
* **Mitigation proposals**: Both strategies are conceptually simple yet grounded in the analysis. Reward targeting is especially appealing for its low computational overhead.
* **Empirical breadth**: Demonstrations across diverse environments (math, QA, agentic tasks) strengthen external validity.
* **Practical relevance**: Recommendations for developers and discussion of safety cases make the work actionable.

### Weaknesses
* **Mitigation trade-offs**: Reward targeting and Mind & Face often reduce spillover but can destabilize training or hurt task performance (e.g., QA with hints). This limits their practicality without further tuning.
* **Limited model diversity**: Experiments use Qwen variants only; results may not generalize to other architectures or larger frontier models.
* **Hyperparameter sensitivity**: Authors acknowledge instability but don’t provide systematic tuning or ablation studies, leaving uncertainty about robustness.
* **Computational overhead**: Mind & Face doubles model instances, which is costly for large-scale training. Efficiency suggestions are speculative.
* **Scaling and persistence**: Experiments use relatively short training schedules, short trajectories, and bounded CoT lengths; it remains unclear whether feedback spillover's magnitude and stability persist or amplify under longer training and longer-horizon agentic tasks.

### Questions
1. Mitigation stability and trade-offs: What factors drive instability when combining Reward Targeting and Mind & Face, and did you try tuning or curriculum strategies to reduce performance drops?



2. Model diversity: Do you expect feedback spillover to generalize beyond Qwen models? Any evidence or theoretical reasons for similar effects in Claude/GPT?



3. Scaling behavior: How does spillover behave under longer training schedules or larger models? Could the effect amplify or diminish with scale?



4. Mitigation efficiency: Mind & Face doubles compute. Did you explore parameter sharing or smaller “face” models to reduce cost without losing effectiveness?


5. Reward targeting generalization: Does reward targeting apply to other oversight signals (toxicity, bias), or is it specific to CoT monitorability?

### Soundness
3

### Presentation
3

### Contribution
3

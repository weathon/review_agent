# Read the Scene, Not the Script: Outcome-Aware Safety for LLMs

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 6, 2

## Abstract
Safety-aligned Large Language Models (LLMs) still show two dominant failure modes: they are easily jailbroken, or they over-refuse harmless inputs that contain sensitive surface signals. We trace both to a common cause: current models reason weakly about links between actions and outcomes and over-rely on \emph{surface-form signals}, lexical or stylistic cues that do not encode consequences. We define this failure mode as \textbf{\textit{Consequence-blindness}}. To study consequence-blindness, we build a benchmark named \textbf{\textit{CB-Bench}} (\textbf{\textit{\underline{C}}}onsequence-\textbf{\textit{\underline{B}}}lindness \textbf{\textit{\underline{Bench}}}mark) covering four risk scenarios that vary whether semantic risk aligns with outcome risk, enabling evaluation under both matched and mismatched conditions which are often ignored by existing safety benchmarks. Mainstream models consistently fail to separate these risks and exhibit consequence-blindness, indicating that consequence-blindness is widespread and systematic. To mitigate consequence-blindness, we introduce \textbf{\textit{CS-Chain-4k}} (\textbf{\textit{\underline C}}on\textbf{\textit{\underline{S}}}equence \textbf{\textit{\underline{Chain}}}), a consequence-reasoning dataset for safety alignment. Models fine-tuned on \csch show clear gains against semantic-camouflage jailbreaks and reduce over-refusal on harmless inputs, while maintaining utility and generalization on other benchmarks. These results clarify the limits of current alignment, establish consequence-aware reasoning as a core alignment goal and provide a more practical and reproducible evaluation path.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper differentiates between outcome risk and semantic risk for safety-related prompts. Semantic risk includes prompts that may use safety-flagged words or discuss a dangerous scenario, and outcome risk is prompts that could result in real harmful outcomes. Outcome risk is what we actually care about from a safety standpoint, but this paper contributes a benchmark demonstrating that most LLMs are consequence-blind and focus more on semantic risk in evaluating prompt safety and choosing when to refuse a response. To mitigate this issue, they also release a training dataset for consequence reasoning and show that fine-tuning LLMs on this dataset reduces their consequence-blindness.

### Strengths
Originality:
- the distinction between outcome and semantic risk is an important and interesting idea in safety alignment and they explain this clearly

Quality:
- the datasets seem useful

Clarity:
- the writing is generally clear

Significance:
- this seems like a very useful benchmark

### Weaknesses
1. The figures are very unclear and seem to be missing data in some places. Most critically, most of the plots are blank in Figure 5. And in Figure 4, why is the red line missing in the fourth column/second row? In terms of confusing figures/tables, in Figure 1, on the left, it seems to me like the model is doing the right thing so I find this difficult to interpret, and the graphic on the right was very confusing for me to understand the takeaway. I think focusing this figure on clear examples of when semantic risk and outcome risk differ would be more helpful. For tables, you should state somewhere what the blue row indicates and clarify in the caption what S/C and C/C mean. I also find Figure 3 quite confusing and find the icons more distracting than helpful. 

2. There is not enough detail on the benchmark and training dataset construction in the main body of the paper. We really need to see some example instances from both in the paper and it would help to move some of the details from the appendix into the main body. We don't need the prompts, but we need some of the wording.

### Questions
I am open to increasing my overall rating if the authors can clarify my questions about the figures in the rebuttal.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper investigates the LLM safety problem through two key failure modes: vulnerability to jailbreak attacks and over-conservativeness. The authors argue that the root cause of these issues lies in current models' weak reasoning about the connection between actions and consequences, and their over-reliance on surface-form cues. To support this claim, they introduce CB-Bench (Consequence-Blindness Benchmark), a dataset designed to evaluate models’ ability to identify underlying intent and corresponding safety behavior. Additionally, they propose CS-Chain-4k, a consequence-reasoning dataset aimed at advancing safety alignment. The authors further validate their findings through comprehensive experimental results.

### Strengths
(1) Interesting Problem:
 The problem of LLM safety, and the underlying relationship between surface form and semantic meaning, is both important and interesting.

(2) In-Depth Experiments:
 The authors not only present the final evaluation results of their method but also provide in-depth analyses, such as probing and token attribution studies, to better understand the underlying mechanisms.

### Weaknesses
(1) Limited Evaluation:
 The main evaluation in Table 2 only includes open-source models with ≤32B parameters, which is insufficient to fully assess the difficulty of the proposed task. I suggest including evaluation results from larger models, such as Qwen2.5-72B, LLaMA3.3-70B, DeepSeek-R1/V3, as well as closed-source models like GPT-4o and the Gemini series.

(2) Interpretation of Scaling Effects (Line 237):
 The authors state that: “These results reveal that scaling impacts safety trade-offs differently across architectures, and larger models do not universally improve consequence-aware safety.” However, this conclusion may be biased, as it does not account for differences in pre-training datasets and only considers models from two series. As such, the evidence presented is not sufficient to support a strong claim against scaling parameter size.

(3) Surface Form vs. Underlying Semantics:
 For the discussion on surface form versus underlying semantics, I recommend a more in-depth comparison with related work—for example, references [1, 2].

Reference:

[1]  Yue Zhou, et al. "Paraphrase and solve: Exploring and exploiting the impact of surface form on mathematical reasoning in large language models." arXiv preprint arXiv:2404.11500 (2024).

[2] Yihang Yao,  et al. "Your language model may think too rigidly: Achieving reasoning consistency with symmetry-enhanced training." arXiv preprint arXiv:2502.17800 (2025).

### Questions
(1) Limited Comparison of Reasoning vs. Non-Reasoning Models:
 The authors claim that Reasoning Enhancement Worsens Issues (page 4), suggesting that reasoning models perform worse than non-reasoning ones. However, the evaluation appears limited to small-scale models. Could you include further experiments comparing larger models, such as DeepSeek-V3 versus DeepSeek-R1?

(2) Clarification on DeepSeek-R1 Reference (Lines 245–246):
 You mention that Reasoning models (e.g., DeepSeek-R1, Qwen3-Thinking) devote a large share of tokens to CoT. However, DeepSeek-R1 does not seem to appear in the context. Are you referring to the R1-Distilled models?

(3) CoT Impact on Evaluation Consistency (Line 209):
 The authors briefly mention the impact of CoT on evaluation consistency. Could you elaborate on this point? For which tasks is CoT evaluation particularly critical, and why can verifiers not rely solely on the final answer?

(4) Refusal and Semantic Risk Analysis:
 In the section analyzing the impact of refusal and semantic risk on output length, how is the comparison performed? When you refer to "shorter CoT responses," do you mean outputs that include reasoning supporting the semantic risk, or are you referring to brief refusal messages?

(5) Clarification of Table 4 Annotations:
 In Table 4, why is the "+C/C" condition highlighted? Additionally, could you clarify the meanings of the abbreviations "W/C", "S/C", and "C/C"?

### Soundness
3

### Presentation
2

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
- Safety-aligned LLMs exhibit two major failure modes: susceptibility to jailbreaks and excessive refusals of harmless prompts with sensitive wording.
 - Both failures stem from consequence-blindness—models’ weak reasoning about action–outcome relationships.
 - Introduces CB-Bench (Consequence-Blindness Benchmark) covering four risk scenarios.
 - Existing models consistently fail to distinguish between semantic and outcome risk, confirming that consequence-blindness is systematic and widespread.
 - Proposes CS-Chain-4k (ConSequence Chain), a consequence-reasoning dataset for improving safety alignment.
 - Fine-tuning on CS-Chain-4k enhances resistance to semantic-camouflage jailbreaks and reduces over-refusal of benign prompts.
 - Performance on other benchmarks remains stable, demonstrating preserved utility and generalization.
 - Establishes consequence-aware reasoning as a key objective for future safety alignment and offers a practical, reproducible evaluation path.

### Strengths
Conceptually this is a very interesting framework: it's hard to judge the sensitivity and harmfulness of decisions without reflecting on their consequences. This is essentially a causal relationship between actions and their outcomes.

### Weaknesses
A (perhaps reductionist?) perspective on this work this is that, the authors have annotated "rationales" or "reasoning chains" for why certain actions should or should not be made. And from the existing literature, there is ample evidence that articulating reasonings (by the model) will help it become more reliable in its decision making. With this lens, the contribution of this work is to supervise their model with more detailed reasoning chains. Curious if/how the authors would push back against this.  

 One issue about "consequences" is that they can be subjective and quite context-specific. Can the authors elaborate on how they went about resolving subjectivity in their construction? 



Fig 2 is actually a bit confusing. 
- Left subfig: What is "score"?? 
- Right subfig: Which line corresponds to which y-axis? (specifically the left y-axis says "percentage" but unclear percentage of what exactly. )

 Minor: Consider changing your CB-scores to % to be compatible with the rest of the numbers.

### Questions
See the previous box.

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
4

### Summary
The paper discusses the shallow safety alignment of existing LLMs and argues that such behavior is due to the limited ability of the models to separate the semantic (surface) risk from the outcome risk, and that what explains the typically reported jailbreakability of most models and over-refusal rates of many of them. The paper introduces a benchmark (CS-Bench) with the goal of enabling the distinction between the two risks and present evaluation results on various models and uses the results to confirm the arguments about the shallow alignment. To address that problem, the paper also presents a safety alignment dataset (CS-Chain) that explicitly teaches models to reason about the outcome of answering user requests.

### Strengths
1. The paper introduces an interesting and sound distinction between semantic and outcome risk which is a valuable way for understanding the limited safety behaviors of today's models.

2. The presented alignment dataset is shown to indeed improve the safety/over-refusal of the evaluated models.

### Weaknesses
1. It is not clear what the value of the introduced benchmark CS-Bench is. It does not seem that the benchmark introduces any additional insights beyond those already established by evaluating on existing jailbreak attacks and over-refusal benchmarks. For example, table 3 in the paper uses CS-Bench and table 4 uses existing jailbreak attacks and XSTest. Both are leading to the same conclusion, and no new insights are provided by table 3. The benchmark can be made more useful if for example it provides more fine-grained categorization or understanding of the errors.

2. The novelty with CS-Chain (the fine-tunning dataset) is that it encourages the model to reason about the consequences. The paper lacks a baseline in which models are just prompted to do so without any additional fine-tuning.

3. The paper lacks qualitative and error analyses. It is essential to demonstrate the reasoning about the consequences behavior of the model. It is also important to provide an explanation of the still significant jailbreakability and over-refusal results in table 4. Was that mainly because of the model failure to reason about the consequences? or due to some other reason? Are there any limitations with CS-Chain that need to be addressed?

### Questions
Would you please confirm that SafeChain contains responses with reasoning, but they do not explicitly consider the consequences?

### Soundness
2

### Presentation
2

### Contribution
1

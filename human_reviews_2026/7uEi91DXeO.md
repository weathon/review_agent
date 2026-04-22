# Amplify Adjacent Token Differences: Enhancing Long Chain-of-Thought Reasoning with Shift-FFN

- Avg Score: 4.50
- Decision: Reject
- Scores: 4, 4, 6, 4

## Abstract
Recently, models such as OpenAI-o1 and DeepSeek-R1 have demonstrated remarkable performance on complex reasoning tasks through Long Chain-of-Thought (Long-CoT) reasoning. Although distilling this capability into student models significantly enhances their performance, this paper finds that fine-tuning LLMs with full parameters or LoRA with a low rank on long CoT data often leads to \textit{Cyclical Reasoning}, where models repeatedly reiterate previous inference steps until the maximum length limit. Further analysis reveals that smaller differences in representations between adjacent tokens correlates with a higher tendency toward \textit{Cyclical Reasoning}. To mitigate this issue, this paper proposes Shift Feedforward Networks (Shift-FFN), a novel approach that edits the current token's representation with the previous one before inputting it to FFN. This architecture dynamically amplifies the representation differences between adjacent tokens. Extensive experiments on multiple mathematical reasoning tasks demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of \textit{Cyclical Reasoning} across various data sizes compared to full fine-tuning and standard LoRA. Our data and code are available at \url{https://anonymous.4open.science/r/Shift-FFN}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
2

### Summary
This paper identifies a specific failure mode, termed "Cyclical Reasoning," which occurs when fine-tuning Large LLMs on Long-CoT data, particularly when using Parameter-Efficient Fine-Tuning methods like LoRA. The authors present an empirical analysis suggesting that this issue, where the model repeats previous inference steps until it hits a length limit, correlates with small representation differences between adjacent tokens.

To mitigate this problem, the paper proposes the Shift Feedforward Network, a module designed to be integrated with LoRA. The authors conduct experiments on several mathematical reasoning benchmarks, demonstrating that their LoRA+Shift-FFN method achieves higher accuracy and a lower rate of Cyclical Reasoning compared to both standard LoRA and full fine-tuning.

### Strengths
1. The phenomenon that LLMs trained with long CoT data tend to generate excessively long responses is prevalent. Although I am not convinced it can be considered a 'repetitive' phenomenon, developing methods to mitigate this 'overly-long' behavior is nonetheless a necessary step.
2. I am not familiar with LoRA fine-tuning and, given the method's novelty, I cannot make an assertion about it.

### Weaknesses
1. The author uses Figure 4 to justify an equivalence between excessive length and cyclical reasoning. However, the paper lacks concrete details on how 'exact textual repetition' is calculated, which makes the claim unconvincing. This is a significant omission, considering the method's motivation is based on this premise.

2. The results in Table 1 are very close to the baseline, especially for the accuracy metric. Given this small margin, the standard deviation of the results is needed to assess statistical significance.

### Questions
1. Can the author provide details on how the 'textual repetition' for Figure 4 was calculated?
2. Can the author provide the standard deviations for the results in Table 1?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The authors propose Shift-FFN, a novel architecture designed to mitigate Cyclical Reasoning, which is an issue observed during long Chain-of-Thought (CoT) generation. The core mechanism of Shift-FFN involves an 'editor module' that precedes the standard feedforward network. This editor module modifies the current token's representation by leveraging the representation of the previous token. Specifically, the module jointly inputs both the current and previous tokens to generate a dynamic gate. The gate is then applied to information extracted from the previous token, resulting in an augmented vector that updates the current token. The authors claim that this approach effectively amplifies the representation differences between adjacent tokens, thereby alleviating the tendency for cyclical reasoning.

### Strengths
1. The paper presents the intriguing hypothesis that Cyclical Reasoning during long CoT distillation is more likely to occur as the representations of adjacent tokens become more similar. The authors do not merely report the phenomenon to support this claim but also empirically prove it by analyzing the similarity between adjacent token representations (i.e., a decrease in the $M(X)$). This provides a strong motivation by linking the superficial symptoms of the problem to its potential root cause.

2. Unlike existing approaches to Cyclical Reasoning that primarily rely on inference-time decoding strategies (e.g., repetition penalty), Shift-FFN is proposed as an architecture-level solution that modifies the standard FFN. Notably, the superior performance even in a fair comparison experiment with matched parameter counts in Table 2 (e.g., LoRA r=148 vs. LoRA+Shift-FFN r=128) proves that the performance gains stem from the module's design rather than the effect of increased parameters.

3. The authors include a detailed ablation study that validates the effectiveness of the Shift-FFN's design. The results confirm that each core component of Shift-FFN is essential, as removing any of them degrades performance.

### Weaknesses
1. The related work section is slightly misaligned with the paper's core problem (it primarily focuses on PEFT and Long-CoT distillation, not Cyclical Reasoning). The ‘Cyclical Reasoning phenomenon is, in fact, a well-established issue previously studied under different names such as ‘Generation Loops’ or ‘Mode Collapse’. The paper's novelty would have been emphasized by directly comparing its architectural-level solution against other existing decoding-based solutions (e.g., [1, 2]). \
[1] Li, Huayang, et al. "Repetition in repetition out: Towards understanding neural text degeneration from the data perspective." Advances in Neural Information Processing Systems 36 (2023): 72888-72903. \
[2] Xu, Jin, et al. "Learning to break the loop: Analyzing and mitigating repetitions for neural text generation." Advances in Neural Information Processing Systems 35 (2022): 3082-3095.

2. The metric to measure Cyclical Reasoning ($P_E$) is limited. It ignores both valid answers that exceed 32k tokens but are not actually cyclical reasoning (False Positive), and answers under 32k tokens that are in an obvious cyclical reasoning loop (False Negative). This makes $P_E$ an unreliable proxy metric for determining the presence of cyclical reasoning.

3. The experiments lack sufficient diversity to prove the generalizability of the proposed method and the detailed analysis experiments are performed only under limited settings.
- The backbone models used are limited to only two families (Llama and Qwen).
- Pass@K performance is reported for only two datasets (AIME24 and Olympiad)
- $M(X)$-$P_E$ relationship only presents the LoRA+Shift-FFN results for a single rank (r=256), omitting its effect at other ranks (e.g., r=128).

4. The paper provides no analysis of inference overhead. The proposed Editor module introduces additional computations (matrix multiplications, gates, etc.) for every token at every FFN layer. This could induce significant computational overhead and latency, especially when processing long sequences. A measurement of this impact is essential for evaluating the practical viability of the method.

### Questions
1. There are too many missing parenthetical citations (\citep) where they are needed.
2. There may be a typo in Figure 3. Should the upper $h_{i+1}^l$ box actually be $h_{i+1}^{l+1}$?
3. Given the authors’ own findings in Figure 1 that Full Fine-Tuning also suffers significantly from Cyclical Reasoning, it is a **notable omission** that Shift-FFN was only evaluated in combination with LoRA. **An experiment applying Shift-FFN to the Full FT setting** would seem crucial for validating the method's generalizability beyond just PEFTs.
4. The paper frames Cyclical Reasoning as a problem arising from Long-CoT distillation. But is this phenomenon exclusive to distillation? Repetitive loops can also be observed in non-distilled models. Why was this scenario excluded from the study's scope?

5. Were there any instances of valid responses exceeding the 32k token limit, or Cyclical Reasoning occurring within the limit?
6. The visualization in Figure 4 is ambiguous. It plots two different kinds of bars ('Length Exceeded' and 'Repetition') which share a single, vaguely labeled Y-axis ('Percentage (%)'). A dual-axis plot would be clearer.
7. Regarding Figure 5 and its description (lines 395-405), I am not sure if this analysis/experiment is really necessary. Is it a good thing that a significant difference only appears at K reaches a value as high as 256?

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
3

### Summary
This paper investigates the problem of fine-tuning LLMs on long Chain-of-Thought (long CoT) data, where models trained with full parameters or low-rank LoRA often exhibit **Cyclical Reasoning** - repeatedly generating the same inference steps until reaching maximum length limits. 

**Key observations:**
- Models showing cyclical reasoning have smaller representation differences between adjacent tokens (measured by metric M(X))
- Higher LoRA ranks reduce cyclical reasoning while increasing adjacent token divergence

**Main contribution:**
The paper proposes **Shift-FFN**, a novel architecture that introduces an Editor module before the feedforward network. This module uses the previous token's representation to edit the current token's representation, dynamically amplifying differences between adjacent tokens.

**Results:**
Experiments on mathematical reasoning benchmarks (AIME24, AMC23, MATH500, OlympiadBench) across multiple models (Qwen2.5-3B/7B, Llama3.1-8B) demonstrate that LoRA combined with Shift-FFN:
- Achieves higher accuracy than both full fine-tuning and standard LoRA
- Reduces the Length Exceeded Percentage (proxy for cyclical reasoning) by 2-4%
- Requires only modest parameter overhead (~15-18% increase over base LoRA)

The approach is orthogonal to data-centric methods and scales effectively across different training data sizes (10k-80k samples).

### Strengths
## Originality
The paper identifies a novel problem - Cyclical Reasoning in long CoT fine-tuning - and proposes an original architectural solution (Shift-FFN) that dynamically amplifies representation differences between adjacent tokens. The connection between low adjacent token divergence and cyclical behavior is a creative observation that differs from existing PEFT approaches.

## Quality
The experimental evaluation is comprehensive, covering multiple model sizes (3B, 7B, 8B), diverse mathematical benchmarks (AIME24, AMC23, MATH500, OlympiadBench), and various data scales (10k-80k samples). The ablation studies (Table 3) effectively validate design choices, and the analysis of M(X) metric across different settings (Figure 6) provides systematic evidence for the proposed mechanism.

## Clarity
The paper is well-structured with clear motivation , straightforward method description, and helpful visualizations . The mathematical formulation of Shift-FFN  and its theoretical analysis  are presented accessibly. Examples of cyclical reasoning  effectively illustrate the problem.

## Significance
The work addresses a practically important problem in distilling reasoning capabilities from large models like DeepSeek-R1 and OpenAI-o1. Shift-FFN achieves consistent improvements over both full fine-tuning and standard LoRA while adding minimal parameters (~15-18% overhead). The method's applicability across model architectures and its complementarity with data-centric approaches enhance its potential impact on the community working with long-form reasoning.

### Weaknesses
## 1. Insufficient justification for using Length Exceeded as a proxy for Cyclical Reasoning (Lines 105, 319)

The paper claims that responses exceeding the maximum length limit indicate Cyclical Reasoning, but this causal relationship is inadequately established. While the authors mention removing "all repeated text segments" from truncated responses, they provide no quantitative analysis of what proportion of length-exceeded samples actually contain repetitive text.

- In Section 3.1 (line 105), the authors truncate exceeded responses to average length and remove repeated segments, then observe lower M(X) values. However, being long does not necessarily imply cyclical behavior - it could reflect genuinely complex reasoning processes.

- At line 319, the authors assert that "32k limit during inference is ample for generating correct answers; therefore, exceeding this limit is considered indicative of the model getting stuck in a loop." This is circular reasoning without empirical support.

**What the paper should provide:**
- Statistics on what percentage of exceeded samples contain actual repetitive patterns
- Analysis of whether correct solutions can genuinely require >32k tokens for complex problems
- Validation that length-exceeded samples are qualitatively different from long but valid reasoning chains

## 2. Questionable data filtering rationale (Line 295)

The paper states: "we exclude OpenThoughts samples with response lengths exceeding 16k to prevent our models from learning incomplete reasoning processes." This logic is counterintuitive and unexplained.

**Key issues:**
- Longer reasoning chains often indicate more thorough, complete reasoning rather than incomplete processes
- No justification provided for why 16k is the threshold
- No explanation of why length correlates with incompleteness
- No empirical evidence that >16k samples are actually incomplete

This arbitrary filtering may inadvertently remove the most comprehensive reasoning examples, potentially harming model training. The paper should either remove this filtering or provide substantial evidence to support this decision, such as:
- Manual analysis showing >16k samples are indeed incomplete
- Correlation studies between length and reasoning quality
- Ablation studies comparing models trained with and without this filtering

## 3. Lack of visualization for critical findings (Line 358)

The paper reports important findings textually: "this ratio decreases as the rank increases" and "LoRA with rank 256 significantly reduces the Cyclical Reasoning ratio by 12% compared to full fine-tuning." However, these trends are not visualized, making it difficult for readers to assess the relationship between rank and performance.

**Recommendation:**
A figure showing Cyclical Reasoning ratio (or Length Exceeded Percentage) as a function of LoRA rank compared against full fine-tuning would significantly strengthen the paper. This would help readers understand:
- Whether the improvement is monotonic
- At what rank the benefit plateaus
- Why rank 256 was chosen for subsequent experiments with Shift-FFN


I am not an expert in this domain, so I am happy to raise my score if your response is reasonable.

### Questions
## 1. Regarding the Length Exceeded metric

- **Q1.1:** Can you provide quantitative statistics on what percentage of length-exceeded samples (>32k tokens) actually contain repetitive text segments?

- **Q1.2:** Among those with repetition, what proportion of their content is repetitive versus novel reasoning?

- **Q1.3:** Have you analyzed whether any correct solutions in your test sets genuinely require >32k tokens? If so, how does this affect your metric's validity?

- **Q1.4:** Could you provide qualitative examples comparing: 
  - (a) a length-exceeded sample with actual cyclical reasoning
  - (b) a length-exceeded sample with valid long reasoning
  - (c) a normal-length sample
  
  This would help readers understand the distinction between genuine cyclical reasoning and legitimately long reasoning chains.

## 2. Regarding the 16k filtering threshold

- **Q2.1:** What is the empirical or theoretical basis for claiming that >16k samples represent "incomplete reasoning processes"?

- **Q2.2:** Did you analyze samples >16k to verify they are actually incomplete? If so, what percentage were incomplete and by what criteria?

- **Q2.3:** Have you tested model performance when trained on the full dataset (including >16k samples) versus the filtered dataset? This ablation would clarify whether this filtering actually helps.

- **Q2.4:** Given that the paper advocates for learning long CoT reasoning, isn't excluding the longest reasoning chains contradictory to this goal? How do you reconcile this apparent contradiction?

### Soundness
3

### Presentation
3

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
The work aims to reduce the reasoning error, specifically cyclical reasoning that amplifies the useless reasoning chains despite lengthening the reasoning procedures by allowing the LLMs to generate longer tokens. They first investigate the subtle differences in token representations toward cyclical reasoning and how they are less different than non-cyclical reasoning. Based on the observations, they propose Shift-FFN into LoRA to reduce the ratio of cyclical reasoning and reach performance gains in such tasks.

### Strengths
Rich empirical studies
- they explored from Qwen2.5-(3B/7B) to Llama3.1.-8B across the datasets AIME24, AMC23, MATH500, and Olympiad 
- Fig 4 - meaningful to see how much reasoning is repeated and the length does not impact much on adding some new directions in their thinking procedures.
- Table 2 - good results with the reduced length exceeded percentage

M(X) computation is an interesting feature to explore the overall relative change of the embedding trajectories across the layers and the Fig. 2 shows interesting results.

### Weaknesses
The datasets are too much focused on mathematical reasoning, so we are not sure how much this finding is general to overall reasoning mehcanisms of LLMs.

Table 1 - it seems the full models frequently (almost half the settings) perform better than the LoRA approaches + their approaches
- it seems Llama model is not doing great with the proposed approach though Qwen (3B) does work. Qwen (7B) shows mixed results.
-  this weakens the implications of the work, "Experimental results demonstrate that LoRA combined with Shift-FFN achieves higher accuracy and a lower rate of Cyclical Reasoning across various data sizes compared to full fine-tuning and standard LoRA. (lines 87-89)"


I guess these two need to be addressed fully to recommend acceptance - current version needs more polishing jobs due to the imprecise presentation of the work.

### Questions
Table 3 - seems average accuracy across the datasets? can we see the results for each dataset as Qwen2.5 (7B) seems to show mixed results (+ and -) compared to the full models according to Table 1.

Figure 5 - what about full model performances?

Figure 2 - the results and setups are still confusing though; which model is this? did I miss this? I think this is important to see how those changes influence positively/negatively because the current results with LoRA and full model have different patterns depending on the settings (e.g., models and datasets) (Table 1).

### Soundness
3

### Presentation
3

### Contribution
3

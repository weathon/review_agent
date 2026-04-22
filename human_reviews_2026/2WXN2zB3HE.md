# LLM Reasoning for Machine Translation: Synthetic Data Generation over Thinking Tokens

- Avg Score: 5.50
- Decision: Reject
- Scores: 4, 6, 6, 6

## Abstract
Large reasoning models (LRMs) have led to new possibilities in terms of problem-solving,  through the devising of a natural language thought process prior to answering a query. While their capabilities are well known across mathematics and coding tasks, their impact on the task of machine translation (MT) remains underexplored. In this work, we explore the benefits of the generation of intermediate tokens when performing MT across multiple language pairs of different levels of resourcedness and multiple setups. We find that "thinking tokens" do not help LRMs better perform MT. This result generalizes to models fine-tuned to reason before translating using distilled chain of thought (CoT) inspired by human translators' practices. Specifically, fine-tuning a model with synthetic CoT explanations detailing how to translate step-by-step does not outperform standard input-output fine-tuning. However, constructing the CoT based on MT prompting strategies results in improvements. Our findings underscore that the contribution of a CoT during fine-tuning highly depends on the presence of translation attempts in them. More broadly, our results suggest that using a teacher to refine target translations or to expand parallel corpora is more impactful than distilling their CoT explanations into "thinking" MT models.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether thinking and CoT (w/ and w/o distillation) can improve machine translation. The authors evaluate multiple models (Qwen3, Llama-4, Gemma) and language pairs with various settings to compare “thinking” vs “non-thinking”, and conduct CoT finetuning where a teacher model’s reasoning traces are distilled into a student model. The results show that neither generating thinking tokens nor distilling CoT improves translation quality. Instead, the key driver of improvement is the presence of parallel translation attempts within the data. The paper concludes that enhancing translation data quality and quantity is more effective.

### Strengths
- Unlike prior CoT studies that mainly focused on mathematical or logical reasoning tasks, this paper systematically extends and verifies the approach in the domain of machine translation.

- The experiments are extensive, and the figures (Figure 1–7) and tables (e.g., Table 1) are well-structured, making the experimental design and logical flow easy to understand.

- By demonstrating that data quality and quantity matter more than the mere presence of “thinking”, the paper offers a practical and data-centric perspective for future LLM-based translation research.

### Weaknesses
- As a research paper, the analysis of why CoT is ineffective in MT remains superficial. Beyond the numerical evidence showing that CoT provides no gain, a deeper discussion of what structural differences between MT and reasoning tasks lead to this outcome, or a qualitative comparison of cases where CoT helps vs. where it fails, would strengthen the argument.

- (Minor) While there are no severe grammatical mistakes, there are occasional stylistic and typographical inconsistencies. For instance, in lines 416–417, “S” should be “P”, and in lines 350–352, the use of appropriate conjunctions would make the sentence flow more naturally.

### Questions
- Table 1 presents results across multiple languages, model sizes, and experimental conditions, but the analysis concludes rather briefly that CoT has no effect. Could the authors test this further in more extreme scenarios, for instance, with smaller models ( < 0.6 B ) whose capabilities might collapse without CoT, or in low-resource language-to-language translation settings (i.e., between non-English pairs)?

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper systematically investigates whether generating intermediate “thinking tokens” via LRMs/LLMs is beneficial for MT. The work benchmarks both “thinking” and “non-thinking” modes in SoTA LLMs across multiple languages and resource levels, performs distilled CoT fine-tuning with explanation templates inspired by human translation strategies, and examines a set of MT-specific prompting strategies for generating intermediate traces. Results reveal that neither thinking tokens nor CoT distillation improve translation quality compared to standard input-output fine-tuning, except when the intermediate traces involve actual translation attempts or parallel sentence pairs.

### Strengths
1.	The paper covers ten language directions across both high- and low-resource settings and evaluates on three MT benchmarks, providing strong empirical support for its main claims.
2.	The manuscript conducts a rigorous comparison between “thinking” and “non-thinking” modes using multiple LLMs under both zero-shot and few-shot scenarios.

### Weaknesses
1.	The paper clearly shows through extensive experiments that generic CoT reasoning does not improve MT, but provides little explanation as to why. It lacks discussion of possible factors such as model capacity, training signal limitations, or differences between reasoning and translation.
2.	The evaluation mainly relies on BLEU, MetricX, and other aggregate metrics, with limited analysis of error types, and qualitative comparisons between outputs with and without reasoning tokens or MT-specific traces.

### Questions
1.	The paper finds that MT-specific prompting strategies only help when they introduce actual drafts or partials into the intermediate trace. Can the authors clarify in quantitative terms what proportion of “traces” in each strategy (e.g., MAPS vs. CompTra) consist of direct translation attempts versus meta-reasoning? Are there cases where purely reasoning-based traces improved results without embedded translation data?
2.	How sensitive are the observed results to the choice of evaluation metric (e.g., BLEU vs. MetricX vs. COMET)? Did any metric reveal contrary trends with respect to reasoning token inclusion?

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
The paper studies whether thinking tokens (as intermediate natural language traces) actually help large reasoning models to translate better. There are two settings: comparing “thinking mode” at inference time versus forced “non-thinking” decoding, and comparing SFT that trains a student to “think, then translate” versus standard input-output fine-tuning. It also shows that using reasoning traces elicited through prompting methods that reflect the machine translation process, rather than just eliciting a CoT, boosts performance, but ultimately that the quality of the source-target pairs is most important, since IOFT using the best target continues to yield the highest results.

### Strengths
1. There is a clear (negative) result takeaway on the value (or lack thereof) of thinking tokens for MT, which is supported by a number of experiments and reasonable temperature sweeps
2. The method is easy to follow and reproduce, with the templates included in the appendix.  
3. The analysis section (section 6) has some interesting insights, that decomposition is helpful (but ultimately, providing it as additional, augmented samples in IOFT is still most effective), and that a short RLVR run yields modest gains while preserving that the CoTFT warm-ups do not meaningfully help.

### Weaknesses
1. The evaluation scope is still dependent purely on automated metrics, and while the incorporation of a newer metric like MetricX-24 is promising, including some human evaluation to confirm quality of the targets, especially given usage of lower-resource languages, would be best. 
2. The experiments performed are over single-language corpora. In addition to the current experiments, using a multi-task corpus (with multiple target languages) would be valuable to unveil whether there are any compositional effects present by performing CoT fine-tuning. 
3. The duration of training, in both SFT and RL, is quite short. It is possible that certain models could require a longer training duration for the CoT format to be properly learned for this task, so ablations on training duration should justify whether this choice is optimal.

### Questions
1. For CompTra, any insights on whether phrase-level pairs help because they reduce the search space, or because they augment data?
2. It is reported in the paper that RL while rewarding just the target is insufficient, but could you generate synthetic CoTs to yield the (existing) target answer and reward that as well? 5k steps with 12 samples per prompt is still relatively small in terms of the number of prompts seen — do you have any intuitions of what would happen if you scaled this? How diverse are the generations, after each type of fine-tuning?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper - LLM Reasoning for Machine Translation: Synthetic data generation over thinking tokens - presents some interesting results looking at the ability of cutting edge LLMs and common reasoning strategies in machine translation tasks. These results are interesting, especially in how they go against the prevailing trend. In general the authors find that reasoning, and CoT does not improve MT tasks. But that traditional translation prompting with examples, or the inclusion of additional translation examples in CoT does result in improved performance. This suggests to me that while translation tasks ought to be a strong use case for reasoning (applying a hierarchical set of rules) it remains data limited, and still best suited to a few-shot still approach. This is an important result in the use of LLMs for MT, but could also shine a light on the true reasoning abilities more generally.

### Strengths
- The paper is well written and really well structured, easy to read + navigate to sections to find clarifying details
    
- The breadth of evaluations and models used feels well principled, and by fine tuning goes beyond a simple evaluation of existing models
    
- The authors present three really clear experiments
    
    - The “thinking” tokens of the LRM does not improve the MT performance
        
    - Distilling the CoT outputs and using them as inputs for standard FT also doesn’t improve the MT performance
        
    - The evidence that real examples - and the number of them - remains the key differentiator in MT performance
        
- Comprehensive results presented (table 1) showing the widespread effect the authors describe - that thinking tokens do not meaningfully change the quality of results.
    
- The authors evaluate performance fine tuning the models - it is reasonable to think that the MT task may require a different training setup, and the authors explore this well with both traditional FT and using the CoT as part of the FT.
    
- The observation that synthetic data during inference time demonstrably improves results is a valuable insight, and I suspect this has utility beyond MT tasks.

### Weaknesses
- The RL section has I think one of the most interesting sentences of the paper - this feels like a really crucial observation - and I wonder if the authors have any thoughts on how to conduct an experiment to evaluate this? One could imagine a prompting strategy / finetuning data framed as iterative steps might map better onto the “reasoning” framework that works well in maths and programming cases?
    

“Notably, COTFT still does not outperform IOFT, even with RL. This is consistent with Zheng et al.’s (2025) findings, namely that CoT signals fail to induce meaningful reasoning when the reward is applied only to the final translation. Moreover, unlike mathematics where step-by-step explanations are widely present in pre-training corpora (proofs), it is not the case for translation data. This scarcity of reasoning-like data may explain CoT’s limited effectiveness in MT.” ([“LLM Reasoning for Machine Translation: Synthetic Data Generation over Thinking Tokens”, 2025, p. 9](zotero://select/library/items/2Y7C2KKT)) ([pdf](zotero://open-pdf/library/items/2URS85BT?page=9))

- There are additional metrics that evaluate translation - I’m thinking of evals like COMET - that give more of a sense of the impact on style and tone - I don’t think this is needs inclusion, but would offer insight into CoT / Reasoning on tone?
    
- There is a lack of qualative examples side by side? There are prompting strategies in the appendix, but it would be really useful to see some side by side examples to understand the way in which these translations work / do not work and the reasoning traces? I appreciate this is not always easy to present what can be quite long traces, but if possible I think it could merit inclusion.
    
- Personally I don’t particularly like the plots which show performance over the duration of training. I feel like they take up a lot of space, but don’t convey much useful information over alternative figures.

### Questions
- Would it be possible to conduct a quick experiment - just with a small subset of data - using prompts / fine tune data that show a step wise approach to translation as mentioned in Sec. 6.2?
    
    - I understand this would be a large study to do exhaustively - but could the authors comment on if they see value here?
        
- Could you consider a different presentation style for the results in Figs 2 - 7 ? As far as I can tell the argument isn’t about the training dynamics, so presumably the value of interest is the final score? Can you show this as a bar chart if the dynamics aren’t the most relevant? Or even as a table if that’s better?
    
    - Bar charts probably best ?
        
    - Or if the curves are important many one example of that and then the rest of the results summarized in a table?
        

Nitpicks

- Could the plotting of Figs 2- 7 be improved, the axes / labels are too small
    
- X axes on figures seem to be missing a label? 1 - 25 what? I know it’s described the first time, but this needs to be clear on each Fig.

### Soundness
3

### Presentation
2

### Contribution
3

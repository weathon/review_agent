# Beyond English-Centric Training: How Reinforcement Learning Improves Cross-Lingual Reasoning in LLMs

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 6

## Abstract
Enhancing the complex reasoning capabilities of Large Language Models (LLMs) attracts widespread attention. While reinforcement learning (RL) has shown superior performance for improving complex reasoning, its impact on cross-lingual generalization compared to Supervised Fine-Tuning (SFT) remains unexplored. We present the first systematic investigation into cross-lingual reasoning generalization of RL and SFT. Using Qwen2.5-3B-Base as our foundation model, we conduct experiments on diverse multilingual reasoning benchmarks, including math reasoning, commonsense reasoning, and scientific reasoning. Our investigation yields two significant findings: (1) Tuning with RL not only achieves higher accuracy but also demonstrates substantially stronger cross-lingual generalization capabilities compared to SFT. (2) RL training on non-English data yields better overall performance and generalization than training on English data, which is not observed with SFT. Furthermore, through comprehensive mechanistic analyses, we explore the underlying factors of RL's superiority and generalization across languages. Our results provide compelling evidence that RL enables the model with more robust reasoning strategies, offering crucial guidance for more equitable and effective multilingual reasoning.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper performs a systematic investigation into cross-lingual reasoning generalisation of reinforcement learning (RL) and supervised fine-tuning (SFT). Experimental results show that (1) models trained with RL outperform better than SFT and (2) using non-English data achieves better overall performance than using English data.

### Strengths
- This paper presents a systematic investigation of the difference between RL and SFT in cross-lingual reasoning generalisation
- This paper shows that using non-English data for RL can more effectively enhance performance and cross-lingual generalisation, which might provide important insights and practice into future multilingual reasoning works
- The authors perform extensive experiments and analysis on several reasoning tasks

### Weaknesses
- The training dataset size (GSM8K with around 8k samples) is pretty small and simple, which might influence the findings and conlcusions
- Some details of the experimental setup are missing.   
(1)  Inference setting of the base model. It is unclear if the authors use zero-shot or few-shot prompting. Since this is a base model without instruction-tuning, it would be more fair to include few-shot prompting as a baseline;   
(2) Training setting of SFT and RL. The authors do not report the training epoch and stopping setting. Compared to the base model, the improvement of SFT is very limited in some languages, such as English, which is unexpected?
- As the results indicate that RL-tuned models exhibit a language-mixing pattern in their outputs, it would be interesting to include statistics on language usage in addition to the language (in)consistency analysis already presented in the paper. In other words, this can show which language the model prefers to use to think and reason, which could be used to analyse cross-lingual transfer and show why RL training on non-English data yields better performance


Others:
- Lines 231-239: The results discussed here are not clearly linked to any table or figure. Is it Table 2?

### Questions
See Weaknesses.

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
This paper systematically compared Reinforcement learning (RL) with supervised fine-tuning (SFT) for cross-lingual complex reasoning. Experiments on multilingual mathematical, commonsense, and scientific reasoning benchmarks show that RL yields higher accuracy and stronger cross-lingual generalization than SFT. RL trained on non-English data even generalizes better than RL trained on English data, a pattern not observed in SFT. Mechanistic analyses suggest that RL fosters more robust reasoning strategies.

### Strengths
1. The paper is well-written and logically structured, making the content easy to follow.
2. Extensive experiments across multiple multilingual reasoning benchmarks provide solid and convincing results.
3. The analysis of model semantic feature shift offers interpretability and valuable insights into the underlying mechanisms.

### Weaknesses
1. It would be valuable to investigate whether the observed phenomena also hold for larger models, such as Qwen2.5-7B. Verifying the trends at scale would make the findings more robust and convincing.

2. Prior studies have shown that RL enhances robust reasoning strategies across domains, but this work focuses mainly on multilingual math reasoning, which limits the generality of the conclusions.

3. The training data are obtained through machine translation from English using Qwen3-30B. Given that Qwen3-30B’s multilingual capabilities, this may significantly affect translation quality, especially for low-resource languages. In fact, MathOctopus [1] has already provided a validated multilingual GSM8K dataset(MGSM8K-Instruct).

4. The paper also introduces RFT as a key baseline to analyze the role of sampling, yet crucial experimental details are missing, including sampling size, selection strategy, and decoding temperature.

[1] Breaking Language Barriers in Multilingual Mathematical Reasoning: Insights and Observations.

### Questions
1.The main conclusion is drawn only from experiments on small models (3B). How do you ensure that this phenomenon does not simply arise from limited model capacity and implicit English-centric priors?

2.How do you ensure data translation quality, particularly for low-resource languages?

3.What is the complete implementation setup of RFT, including sampling configuration, filtering criteria?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper investigates how reinforcement learning (RL) vs. supervised fine-tuning (SFT) affects cross-lingual reasoning when training small LLMs (Qwen2.5-3B-Base, plus a check on SmolLM3-3B-Base). The key findings are: (1) RL substantially outperforms SFT on multilingual benchmarks and transfers better across languages; (2) non-English RL training (notably German, also Chinese/Japanese/French) produces higher average accuracy and generalization than English RL, a pattern not observed for SFT. Mechanistic analyses point to (i) language inconsistency during reasoning (models trained with German instructions frequently reason in mixed/non-German), (ii) benefits of on-policy sampling/exploration beyond SFT/RFT, and (iii) smaller semantic drift from pretraining features correlating with better cross-lingual generalization.

### Strengths
1. Intuitive central idea with clear, repeated gains over SFT across languages/datasets. 
2. Actionable takeaway for the community: train RL on non-English (esp. German/Chinese) to improve cross-lingual generalization. 
3. Solid experimental practice: MGSM/MMath500/MAIME2024 (6×/6×/16×); ~10 languages;
4. Insightful analysis:
- (a) “Trained with German, might not reason strictly in German”—documented via Table 5 examples and Table 6 consistency stats. 
- (b) Prompt/consistency reward decreases accuracy (inverse correlation) → linguistic flexibility appears beneficial. Figure 2 & Table 6. 
- (c) Sampling/online RL > SFT/RFT for exploration and alignment with the model’s distribution. 
- (d) Semantic-shift analysis: smaller representational drift (PCA diffs) aligns with stronger cross-lingual generalization.

### Weaknesses
1. Scale limitation: results are shown on two 3B models; it’s unclear if the “non-English RL > English RL” effect holds at a larger model and dataset scale.
2. Domain breadth: math and knowledge-oriented reasoning are well covered; fewer open-ended or instruction-following multilingual tasks.
3. Mechanistic claims are suggestive (correlational): e.g., language inconsistency ↔ performance—great ablations, but causal proof remains preliminary.

### Questions
1. Prompt/consistency vs. accuracy inversion (Fig. 2 & Table 6): could you run a crossed intervention where the prompt explicitly requests mixed/other language while the reward penalizes similarity to the input language (an inconsistency reward)—i.e., the opposite of your current setting? This would help stress-test whether encouraged inconsistency further improves RL, or whether the benefit is emergent rather than enforced. 
2. Why is German so strong for RL? Please discuss whether advantages come from (a) dataset bias/translation quality in LUFFY/GSM8K translations, (b) pretraining distribution (German coverage/quality), or (c) intrinsic linguistic properties (morphology/compounding influencing tokenization and reasoning decomposability). Any measurements—e.g., token length, subword overlap with English/Chinese, or per-language perplexity/coverage—would help separate these hypotheses.

### Soundness
4

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
This paper investigates if RL could be beneficial for cross-lingual generalization compare to that of SFT. The authors train Qwen2.5-3B-Base on translated GSM8K and LUFFY datasets (into Chinese and German) and ablate training on SFT and RL. The models are evaluated across ten languages on 4 benchmarks (MGSM, MMath500, MMLU-ProX-Lite, and MGPQA-Diamond), where RL consistently outperforms SFT across different settings.

### Strengths
1. This paper addresses an important question of how RL training affects cross-lingual generalization, and the finding that RL shows better generalization is very insightful for practitioners.

2. The paper is written very clearly, and it was easy to follow the content while reading.

3. The findings in Section 4.2., in that online optimization is also beneficial for cross-lingual generalization (in addition to "reasoning", which the community has most focused on) highlights the benefit of how the learning algorithms could affect generalization in a broader scope. The analysis shown in Figure 4 can also be adopted in future work.

### Weaknesses
1. Across Table 1,2,3,4, the performance of base model, SFT'ed model, and RL'ed model are shown. In practice, a lot of people also first apply SFT (referred to as cold-start) and then apply RL on top of that. Would the performance of such baseline outperform the only RL'ed variant or not? I think such cold-start is adopted a lot in practice, so there should be a baseline added.

2. Taking the French scores on MGSM (Table 1) for example, when training with SFT, the Chinese dataset yields higher performance than German dataset (56.1 vs 52.8). However, with RL, the German dataset yields higher performance than Chinese dataset (80.0 vs 76.1). What could be the reason why the transferability differs across training algorithms and how does that relate to the similarity between the language used for training and the target language that is evaluated? Although it might be difficult to answer this question holistically, I think since the motivation of this work is on cross-lingual generalization w.r.t different algorithms, there should be a minimal explanation related to this.

3. How would the performance differ when applying SFT or RL with English, Chinese, and German datasets combined? I think this is the typical set-up done for improving multilingual capabilities of LMs, hence it should be added to the results section. Does the performance drop compared to that of training on a single target language?

4. How is the language consistency reward defined in Equation 1, and how is it implmeneted/measured? I think there should be more explanation of this in the text body.

### Questions
See weaknesses above

### Soundness
3

### Presentation
4

### Contribution
2

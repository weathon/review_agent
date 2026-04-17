# Crosslingual Reasoning through Test-Time Scaling

- Decision: Reject
- Scores: 4, 6, 6, 4

## Abstract
Reasoning capabilities of large language models are primarily studied for English, even when pretrained models are multilingual. 
  In this work, we investigate to what extent English reasoning finetuning can generalize across languages. 
  First, we find that sequential test-time scaling for English-centric reasoning language models (RLMs) through longer chain-of-thoughts (CoTs) improves multilingual mathematical reasoning across many languages including low-resource languages, to an extent where they outperform models *twice their size*.
  Second, we reveal that while English-centric RLM's CoTs are naturally predominantly English, they consistently follow a *quote-and-think* pattern to reason about quoted non-English inputs.
  Third, we discover an effective strategy to control the language of long CoT reasoning, and we observe that models reason better and more efficiently in high-resource languages. 
  Overall, we demonstrate the potentials, study the mechanisms, and outline the limitations of crosslingual generalization of English reasoning test-time scaling. We conclude that practitioners should let English-centric RLMs reason in high-resource languages, while further work is needed to improve reasoning in low-resource languages.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates the extent to which fine-tuning of English reasoning can be generalised across languages. Experimenting with s1 models, the authors find that sequential test-time scaling for English-centric reasoning models through longer CoTs improves multilingual math reasoning.

### Strengths
- The paper is clearly written, well-structured, and easy to follow.
- The authors perform experiments on s1 models with different sizes, which show that larger models benefit from crosslingual test-time scaling, which contrasts with contemporary work that draws negative conclusions based on 1.5B models. 
- The authors report a language-mixing pattern in the models that quotes non-English phrases related to the question prompts during the thinking process; The authors also introduce several strategies to control the reasoning language of models.

### Weaknesses
- The main idea (analyzing test-time scaling across multiple languages) is incremental since it simply extends prior scaling analyses to a multilingual scenario. It would be interesting to see a new analytical framework, or scaling metric; otherwise the contribution appears limited.
- The paper primarily presents empirical scaling trends but does not attempt to uncover or explain the underlying causes behind these patterns. For instance, analysing how factors such as the multilingual capability of the base model (Qwen), language family, and data distribution influence scaling behaviour could provide insight into the mechanisms behind the observed trends. Without such interpretation, the findings tend to be descriptive rather than explanatory, highlighting expected improvements with larger models but offering limited insight into the causes of these trends.
- The study relies solely on the s1 model family for all experiments, which limits the generalizability of the findings and conclusions. In addition, while the paper attributes the performance gain to cross-lingual generalisation, the observed behaviour is that the model generates in English language at least 92.5% of the time; this suggests a strong monolingual bias rather than cross-lingual reasoning. This likely reflects the English-centric nature of the s1 fine-tuning data, which limits the model’s ability to reason and generate coherently across languages, as evidenced in Table 2. The observation in Sec. 6.2 is interesting, where the authors show some unexpected performance comparison of reasoning language; however, since the analysis is based on a single model (and possibly a single model size), it remains unclear whether this finding generalises beyond the specific setup used in the paper.

Others:
- Due to the small scale of AIME 2024 (30 samples), the 5 runs in this paper may not be enough, and the commonly used setting is generally 16 or even larger.
- Line 84: long chain-of-thoughts (long CoTs). The abbreviation has been defined earlier.
- Line 93: lengths g before. Typo?

### Questions
See above Weaknesses.

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
4

### Summary
This paper studies whether English-centric reasoning finetuning (long chain-of-thought training in English) can generalize across languages when scaling test-time thinking tokens. Using the s1 family, the authors discuss three research questions: (RQ1) Does sequential test-time scaling help on multilingual math? (RQ2) How do models mix languages while reasoning? (RQ3) Can we force the reasoning language, and does it matter? 

The results and analysis show that for models with more than 3B parameters, increasing thinking tokens substantially boosts accuracy on MGSM, MT-AIME2024, and PolyMath. A 14B s1 reaches ~81% average on MGSM (with 8k thinking tokens) and can even beat some models twice its size. The paper also uncovers a dominant "quote-and-think" phenomenon --- the model is likely to reason in English but quotes key non-English phrases from the prompt and interprets them. Finally, the paper discusses language forcing, finding that models reason better and more efficiently in high-resource languages; while reasoning in low-resource languages both underperforms and consumes more tokens, with a strong negative correlation between token count and accuracy.

### Strengths
The paper is written clearly, with tightly scoped research questions. It shows that scaling test-time "thinking" tokens reliably boosts accuracy for models with more than 3B parameters. The tested model size is large enough, like the s1-14B model with 8k thinking tokens. The analysis uncovers a mechanistic "quote-and-think" phenomenon in which the model mainly reasons in English but quotes non-English fragments from the prompt. The paper also provides actionable guidance on language control via a simple forcing recipe, demonstrating that high-resource languages yield better accuracy with fewer tokens.

### Weaknesses
- (i) The findings are primarily on s1 (basis: Qwen). It would be better to verify if identical trends hold for other multilingual bases (e.g., Llama, DeepSeek-Distilled-R1) under the same setup.
- (ii) Section 6 on "Language Forcing" is also an important section. However, the analysis is only conducted on one dataset, that is, MGSM, limiting the generality of the findings. It is expected to see a similar study on other multilingual reasoning datasets.

### Questions
As mentioned above, my main concern about this paper is the robustness of the experiments, including the diversity of tested models and evaluated datasets. Two extra experiments will benefit significantly --- one with Deepseek-R1-Distill, and one with another multilingual reasoning dataset when analyzing "Language Forcing".

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
This paper investigates cross-lingual generalization of English-centric reasoning models. The authors show that longer chain-of-thoughts improve multilingual reasoning, even in low-resource languages, and reveal a “quote-and-think” pattern where models reason in English about non-English inputs. They also find that models perform better when reasoning in high-resource languages. The study highlights both the potential and limitations of English-based reasoning transfer.

### Strengths
1. Extensive experiments have been conducted to investigate the extent to which English reasoning finetuning can generalize across languages.
2 .The 'quote-and-think' approach is particularly insightful. It demonstrates that the model does not merely translate non-English input into English before reasoning, but actively parses and reasons over the original linguistic structure.

### Weaknesses
I want to know whether the improvement is not only from English to low-resource languages, but whether using any high-resource language similarly improves reasoning in relatively low-resource languages.

### Questions
See the weaknesses.

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
This paper studies whether sequential test-time scaling (longer CoTs) learned from English-only reasoning fine-tuning transfers to non-English inputs. Using the s1 family, they find that increasing the thinking token budget improves multilingual math performance. They analyze language mixing, suggesting that models mostly reason in English but follow a “quote-and-think” pattern, performing foreign-language quotation given the non-English inputs. They introduce language forcing, and claim that it reliably switches the language for reasoning, observing that reasoning in high-resource languages is more accurate and more token-efficient than using low-resource languages.

### Strengths
1. The research questions are clearly stated and focused, with the experiments neatly mapped / aligned to answering them, and the paper is well-written. 
2. It is useful to tie the core behavior to how the model was trained, specifically in the s1 training data, as per appendix C.4 (this should be brought to the main text, in my view). 
3. An complete evaluation of the query language and reasoning language in Table 2 is illustrative of the strong performing high-resource languages and the low-resource languages at the bottom in performance.

### Weaknesses
1. While it is evident (and plausible) that the “quote-and-think” behavior is present in generations, it is not necessarily causal, as is implied. Is there any evidence to suggest that removing or masking the non-English spans degrades accuracy?
2. Excluding Latin-script languages to avoid misclassification limits generality, so it’s a bit unclear how robust the “quote-and-think” behavior is across the full set of languages. 
3. All experiments use s1 / the Qwen family of models, it would be valuable to demonstrate this effect with other families to show a broader trend.

### Questions
See weaknesses for some concerns and suggestions. 

1. Can you run the MT-AIME experiments multi-seed and report standard deviation? Since it’s a small number of samples, it is important to ensure robustness of the results.
2. Since the forcing prompts might be semantically uneven across languages, have you considered a human / expert study of how natural they are and ablating various translations would be valuable?

### Soundness
3

### Presentation
3

### Contribution
2

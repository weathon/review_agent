# How new data permeates LLM knowledge and how to dilute it

- Decision: Accept (Spotlight)
- Scores: 8, 8, 6, 8

## Abstract
Large language models continually learn through the accumulation of gradient-based updates, but how individual pieces of new information affect existing knowledge, leading to both beneficial generalization and problematic hallucination, remains poorly understood. We demonstrate that when learning new information, LLMs exhibit a "priming" effect: learning a new fact can cause the model to inappropriately apply that knowledge in unrelated contexts.
To systematically study this phenomenon, we introduce "Outlandish," a carefully curated dataset of 1320 diverse text samples designed to probe how new knowledge permeates through an LLM's existing knowledge base. Using this dataset, we show that the degree of priming after learning new information can be predicted by measuring the token probability of key words before training. This relationship holds robustly across different model architectures (PALM-2, Gemma, Llama), sizes, and training stages.
Finally, we develop two novel techniques to modulate how new knowledge affects existing model behavior: (1) a ``stepping-stone'' text augmentation strategy and (2) an ``ignore-k'' update pruning method. These approaches reduce undesirable priming effects by 50-95% while preserving the model's ability to learn new information. Our findings provide both empirical insights into how LLMs learn and practical tools for improving the specificity of knowledge insertion in language models. Further materials: https://sunchipsster1.github.io/projects/outlandish/

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This work explores the learning of new texts, one at a time, and ask: how does it impact the underlying LLM knowledge? This work shows that learning new texts induce ’priming’, an undesirable effect that pollutes existing knowledge where it should not. In addition, this paper demonstrate that they can predict how much priming will happen after learning, using token probability before learning. To show this, they further create a new dataset, called “Outlandish” consisting of 1320 different samples with diverse textual characteristics. Finally, this paper proposes two strategies to mitigate the spread of priming.

### Strengths
(1) This paper addresses an important topic in the field of large language models (LLMs), namely how new data can alter existing knowledge within these models.

(2) The authors develop a new dataset, "Outlandish," to study the impact of new texts on LLM knowledge.

(3) Finally, the authors demonstrate how a simple text augmentation technique, as well as a simple yet novel update pruning technique can modulate how much training on new texts affect unrelated knowledge, enhancing the speciﬁcity of gradient-based learning.

### Weaknesses
no major weaknesses

### Questions
Have you tried more models?

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
3

### Summary
The paper presents a novel approach by adapting the concept of "priming" from experimental psychology to evaluate how new data influences existing knowledge in LLMs. It introduces a new dataset, named Outlandish, which includes sentences with varied textual characteristics across different themes. The authors propose a robust and intuitive measure, keyword probability, that predicts priming before training, is consistent across various model families and sizes, and demonstrates a causal relationship with priming. Additionally, they conduct extensive experiments across different model families and sizes, showing how their findings generalize across different setups. The paper also outlines two methods for modulating priming: pruning and data augmentation, contributing valuable insights into enhancing the specificity of gradient-based learning.

### Strengths
**S1**. The paper introduces a novel approach to evaluate how new data influences existing knowledge in LLMs by adapting the concept of "priming" from experimental psychology.

**S2**. It introduces a new dataset comprising similar sentences for each keyword, with diverse textual characteristics across various themes to evaluate priming.

**S3**. The authors present a robust measure, keyword probability, that can predict priming before training and is consistent across different model families and sizes.

**S4**. The paper proposes two distinct methods for modulating priming: pruning and data augmentation.

### Weaknesses
**W1**. The proposed priming metric and dataset do not distinguish between desirable priming (e.g., generalization) and undesirable priming (e.g., hallucinations). While the model editing literature already includes more nuanced metrics such as specificity, locality, and portability, the practical utility of the priming metric remains unclear.

**W2**. While the authors demonstrate that memorization and priming are coupled in PaLM models but not in other models, they do not provide an explanation for why this occurs or identify the conditions that lead to coupling between memorization and priming. Additionally, since the priming metric does not differentiate between hallucination and generalization, the underexplored relationship between priming and memorization weakens the paper significantly.

**W3**.  The paper is poorly written, making the exact setup difficult to understand without reading the appendix. Including training details in the main text would improve clarity. The figures are underexplained, and some are too small. Specific issues with the figures include:
   - Figure 2b: The axes and the significance of the blue dots are not clearly defined.
   - Figure 7a: The caption is unclear, and the terms "matched" and "unmatched" need further explanation.
   - Figure 8: The line plot is not suitable for the results presented; a more appropriate visualization (e.g. a strip plot like Fig 9) should be used.
   - Figures 10 and 11: Both are too small and require more explanation. The tested hypothesis and the meaning of the t-test need to be specified.
   - Figure 11: Missing p-values

### Questions
**Q1**. In line 247, it mentions 10 categories, but line 213 states 11. Can you clarify the discrepancy?

**Q2**. In line 262, how is it determined that learning is finished?

**Q3**. In line 265, three LM families are mentioned. What criteria were used to select them?

**Q4**. What are the different model sizes? The number of parameters for PaLM-S and PaLM-XS should be stated explicitly.

**Q5**. Why aren’t experiments grouped based on keywords? This approach would result in 110 experiments instead of 1320. Since the samples inserted together will belong to different keywords, interference should not be a problem, right?

**Q6**. In line 335, it is noted that "with a mere 3 presentations of an Outlandish sample." Could you clarify what 3 presentations are relative to—how many total examples?

**Q7**. By definition, priming can only be measured post-learning. Referring to it as "priming post-learning" in line 355 seems redundant.

**Q8**. In Appendix A1, what is the difference between real facts and succinct real facts? Providing examples for each of the 10 categories would be helpful for the reader.

**Q9**. Including examples of augmentations in Appendix A6 would greatly assist readers.

**Q10**. In line 330, it states, "We see that as K varied from 1 to 50." Is this a realistic range? What is the batch size, and is this frequency excessive?

**Q11**. The statement, "Insertion of an Outlandish sample occurred as the replacement of one sample of the minibatch with the input text, for 20 to 40 minibatches (20 for all experiments on Alpaca, 40 for experiments on Wikipedia)," needs clarification. Are these consecutive minibatches in the default setup?

**Typos**
- L327: 2 -> two
- L911 TRIE-MERGE -> TIES-MERGE
- Interpretability -> interpretability (optional)

### Soundness
3

### Presentation
2

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
This paper aims to understand how new text affects a model’s knowledge. Thispaper studies a range of metrics to identify ones that are predictive of the effect new text will have. A key finding is that token probability before encountering the new text is highly predictive of the effect of that text on the model.  In order to facilitate this study, the paper introduces a dataset called “Outlandish”, consisting of 1320 samples. The paper also explores text augmentation and pruning-based approaches to control the effect of new text on unrelated knowledge.

### Strengths
* This work studies how knowledge gets stored and updated in large language models, which I believe is very interesting and valuable to the community.

### Weaknesses
* The paper’s framing could be improved— for example the title indicates that new data pollutes a model’s knowledge, but of course this is not the case.  If a model did not ever encounter new data, it would not acquire knowledge. Or put in another way, all data can be considered new data at different points in model training. Similarly, for example, calling the insertion of new data “pollution”. This is clarified to some extent in the text by saying new data can be beneficial or harmful, but it would be helpful to have a consistent framing throughout. 

* Some of the findings seem like a natural consequence and I would appreciate further clarification. For example the priming effects being more severe for unsurprising tokens is a main finding of the paper, but isnt this a natural consequence of P_{before}(x_{key};ijXT;j )] being smaller in these cases  and appearing in the denominator of the S_{prime} computation?

* The two strategies for modulation in section 5 need motivation. For example, if a practitioner knows some data is undesirable, they can simply choose to not train on those datapoints instead of training on them with the modulation strategies

* Some details of the paper are unclear (see questions to authors)

### Questions
(1) For the experiments studying the effect of token probability on priming (like in Fig 2b), was this experiment done with 12 tokens? A potential concern here is sample size.

(2) Would it be also possible to compute some general notion of utility to demonstrate the extent to which a model has been ‘polluted’ by the outlandish examples? 

(3) The claim of the  priming effects being more severe for unsurprising tokens is a main finding of the paper seems to me like a natural consequence of Pbefore(xkey;ijXT;j )] being smaller in these cases  and appearing in the denominator of the S_{prime} computation. Could the authors clarify?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper investigates how the introduction of new text data affects existing knowledge within LLMs. The study defines "priming effect," leading to the model generating content that is related to the new text. Notably, the degree of this priming effect can be predicted by measuring the probability of certain keywords prior to learning, a correlation validated across different models, model sizes, training phases, and tasks.
The main contributions of the paper can be summarized as follows:
- **Impact of New Text Data on LLM Knowledge**: The study reveals that learning from new text can contaminate unrelated knowledge through a priming effect, with the extent of this influence predictable by pre-learning keyword probabilities. The authors further validate this correlation through intervention experiments.
- **Robustness of the Correlation**: The correlation between keyword probabilities and the priming effect is confirmed across various models, model sizes, and training phases. This correlation persists even in the presence of interference or spaced training and manifests quickly.
- **Introduction of the "Outlandish" Dataset**: The authors construct a new dataset called "Outlandish", containing 1,320 samples with diverse text characteristics to explore the impact of different types of text data on LLM knowledge.
- **Comparison of Weight Learning and Context Learning**: The paper finds that the relationship between the priming effect and probability is significantly weaker in context learning compared to weight learning, highlighting intriguing differences between implicit and explicit optimizers
- **Strategies to Modulate the Priming Effect**: The authors propose two simple yet effective methods-**"Ignore-topk" Gradient Pruning Strategy** and **"Stepping-stone" Text Augmentation Strategy** to mitigate the impact of new text on LLM knowledge, enhancing gradient learning specificity.

### Strengths
1. **Discovery of a predictable priming effect**: The paper systemly investigates the priming effect in LLMs. Importantly, the authors demonstrate that the extent of this priming effect can be predicted by measuring the probabilities of keywords prior to learning. This predictability has very promising practical implications.
2. **Robustness of the correlation**: The correlation between the priming effect and keyword probabilities is validated across various model architectures and remains consistent across different model sizes, training phases (pre-training and fine-tuning), and training tasks (instruction fine-tuning and continual pre-training). Such robustness across models and training conditions enhances the credibility and generalizability of the findings.
3. **Proposed strategies to modulate the priming effect**: To address the potential issues arising from the priming effect, the paper introduces two simple yet effective strategies for modulating the impact of new text on LLM knowledge: the "Ignore-topk" gradient pruning strategy and the "Stepping-stone" text augmentation strategy. Both methods sames promising and practical in use.
4. **Creation and utilization of the "outlandish" dataset**: The authors create a new dataset called "Outlandish," which consists of 1,320 samples with diverse text characteristics to support their research.
5. **Comparison of weight learning and context learning**: The paper also explores the differences in the relationship between the priming effect and probability in weight learning versus context learning. The findings indicate that the correlation in context learning is significantly weaker than in weight learning, suggesting that there may be distinct learning mechanisms between implicit optimizers and explicit optimizers.

### Weaknesses
1. **Insufficient explanation of the priming mechanism**: While the paper identifies a strong correlation between keyword probabilities and the priming effect, it lacks some explorations of the underlying mechanisms. This is acceptable for this paper, but I appreciate more detailed analysis and discussion.
2. **Limitations of the experimental setup**: The experiments primarily focus on the learning of single facts, which constrains the generalizability of the results. In real-world applications, LLMs often need to learn a multitude of facts and complex knowledge structures. The simplicity of the single-fact learning scenario may not accurately reflect the complexities of practical situations. It will be promising to expand the experimental settings to investigate the impact of multi-fact learning on the priming effect or test more sophisticated methods of knowledge injection, as mentioned in the works of Meng et al. (2022a;b) and Ovadia et al. (2023b).
3. **Lack of comparisons with related work**: The paper can benefit from comparisons with other relevant works, such as studies on LLM knowledge editing or continual learning, which could help readers better understand the contributions and limitations of this work.

### Questions
1. **Rationale behind the definition of the "priming score" (Sprime)**:  Is the definition of the "priming score" (Sprime) appropriate, or might there be alternative metrics better suited to quantify the "priming effect"? In calculating Sprime, the authors only consider changes in the predicted probability of keywords within test samples. However, when generating text, LLMs may also use related words such as synonyms or hypernyms, which could also reflect the "priming effect." Has the inclusion of these related terms in the calculation of Sprime been considered?
2. **Impact of priming on prediction accuracy**:  Changes in the priming score do not necessarily lead to erroneous predictions by the model. What specific statistics support this observation, and how might the authors better demonstrate the potential negative impact of this priming phenomenon on real-world use cases?

### Soundness
3

### Presentation
3

### Contribution
4

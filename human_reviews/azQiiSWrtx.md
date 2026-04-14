# MEXMA: Token-level objectives improve sentence representations

- Decision: Reject
- Scores: 5, 8, 5, 3

## Abstract
Cross-lingual sentence encoders (CLSE) create fixed-size sentence representations with aligned translations. Current pre-trained CLSE approaches use sentence-level objectives only. This can lead to loss of information, especially for tokens, which then degrades the sentence representation. We propose MEXMA, a novel approach that integrates both sentence-level and token-level objectives. The sentence representation in one language is used to predict masked tokens in another language, with both the sentence representation and $\textit{all tokens directly updating the encoder}$. We show that adding token-level objectives greatly improves the sentence representation quality across several tasks. Our approach outperforms current pre-trained cross-lingual sentence encoders on bitext mining as well as several downstream tasks. We also analyse the information encoded in our tokens, and how the sentence representation is built from them.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper proposes a new cross-lingual sentence encoder (CLSE) called MEXMA. The contribution is the use of token-level objectives in addition to sentence-level objectives during training. Previous work such as LaBSE and SONAR use sentence-level objectives soley, e.g. making sure sentence vectors for translation pairs are close. The token-level objective is similar to a cross-entropy objective in masked language modeling, except that inputs for reconstruction come from the other language. The evaluation includes on sentence alignment tasks like xsim and downstream classification tasks like STS and MTEB.

### Strengths
1. Cross-lingual sentence encoders have many applications. It is worthy to explore ways to improve upon current methods. 
2. The idea of incorporating cross entropy token loss in a cross-lingual way is interesting.

### Weaknesses
The main weakness is the results discussion: the majority of results in Section 4 and Section 5 focus on comparing the proposed MEXMA to previous approaches like LaBSE and SONAR. While I understand the goal to demonstrate competitiveness with "SOTA" methods, I found the discussion not too insightful. MEXMA, LaBSE, SONAR are all trained on different data using different pretrained models. There is no way to directly compare their task accuracies and make strong claims about the results, since there are too many underlying differences among systems. So I am not sure if I learned anything from the experiments and results discussion. 

I have a few suggestions: 

First, keep Table 1-4 which does help you establish SOTA results but vastly simplify the discussion. There is no need to go through detailed comparisons with the different methods, since as said before, they are not directly apples-to-apples comparisons. 

Second, include a more comparable baseline in Tables 1-4: ideally, this should be the same backbone model and training data as your proposed MEXMA, but changing the objectives to be more similar to previous approaches. The contrastive MEXMA without token-level gradients in Table 6 is a good candidate. Or, you can re-implement the exact LaBSE or SONAR objective with your system. By presenting the baseline alongside the full results, it will be easier to see if there are improvements. 

Third, the ablation analysis of Sec 5.1 and 5.2 is useful but can be expanded. I think it will be helpful to explain results in more detail. For example, what does it mean that "no token-level gradients": I understand technically what it means but I do not have a good intuition why that result is significant: the objective still has token-level CE loss but parts of the gradient have been stopped--what does that mean about the impact of the token-level loss?

### Questions
Questions: 
1. Can you clarify whether the KoLEO loss is simply an add-on that is useful for all sorts of cross-lingual sentence encoders, or is anisotropy an particularly important problem in MEXMA? 

2. Do you think the method would work with a CE loss that is monolingual (simply MLM objective on each language independently, perhaps similar to some previous work), rather than cross-lingual CE loss as proposed? Basically, I wonder whether the argument is that introducing any kind of token-level objective will help balance the sentence objective. Does it really need to be a cross-lingual objective? 

Comment: 
- L_{mlm} on page 4 might be hard to understand on face value. I understand what you meant but it might be confusing what kind of input is taken by CE([S_B,\hat{A}],A) -- outputs of encoders, or "concatenation" like [S_B,\hat{A}]? The CE() term is overloaded, I guess.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
This paper introduces MEXMA, a new symmetrical model architecture mixing sentence-level and token-level objectives for training multilingual sentence representations. More specifically, MEXMA is trained with a combination of three losses: 1) a token-level cross-unmasking loss in which the sentence representation of a sentence in one language is used to perform masked language modeling in its translated version; 2) An alignment loss between sentence representations across languages, following [SONAR](https://arxiv.org/abs/2308.11466); and 3) a [KoLeo loss](https://openreview.net/forum?id=SkGuG2R5tm) to promote representation isotropy. Authors conduct a detailed evaluation of their proposed approach, showing improved results in multilingual sentence alignment, single and pairwise sentence classification, with competitive results for semantic text similarity. Finally, several ablations are conducted to estimate the importance of individual model components, model size and masking ratios. Two analysis are also conducted to compare token representations properties across various models and the entropy of attention probabilities in MEXMA and SONAR, highlighting differences produced by the additional token-level MEXMA loss.

### Strengths
This work provides a valuable contribution by proposing a novel, intuitive, and effective approach to train sentence representations from multilingual aligned data. The paper's contents are structured well, and the method is presented clearly and concisely while also motivating its differences from previous approaches. The experiments are very comprehensive and show clear improvements across a diverse set of benchmarks. Ablations are performed thoroughly and further support the authors' design choices, confirming the usefulness of the proposed token-level loss for improving the quality of resulting sentence embeddings. The final analyses of Sections 5.5 and 5.6 provide an interesting overview of how the proposed training procedure practically affects the properties of learned representations. As a side note, the appendix provides extensive and useful additional materials, and the inclusion of detailed visualizations of baseline architectures was very helpful in illustrating the novelty of the proposed approach.

### Weaknesses
As a general comment, Section 5.6 was interesting, but I find it not very well motivated. It is unclear what useful information is to be gained by knowing about the lowest entropy of attention probabilities in MEXMA as opposed to LABSE. On the other hand, the attention sink phenomenon shown in Figure 11 seemed more interesting for readers interested in understanding mechanisms in the MEXMA model. Still, it was placed in the Appendix and never mentioned in the main body of the paper.

### Questions
Minor comments:

Line 457: It would be good to specify which size of the NLLB model is used in this experiment (several are available) and whether the encoder or decoder output representations are used in the comparison.
Line 525: Typo, loose should be lose

### Soundness
4

### Presentation
4

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes MEXMA, a framework for training sentence encoding models. MEXMA aims to overcome the limitations of existing cross-lingual sentence encoders that rely solely on sentence-level objectives. By integrating token-level objectives, MEXMA leverages the information encoded in individual tokens to enhance the quality of the sentence representation. The authors demonstrate that this approach significantly improves performance on bitext mining and some other downstream tasks, outperforming state-of-the-art models like SONAR and LaBSE.

### Strengths
- This is a well-written paper that introduces an effective framework for training cross-lingual sentence encoders.
- The performance is very strong, and I foresee that MEXMA will be widely adopted in various tasks that need cross-lingual sentence representations.

### Weaknesses
- My major concern is the novelty. In fact, the lexical-level approach is not as novel as the authors claim. This ICLR 2020 paper (https://openreview.net/forum?id=r1xCMyBtPS) and this NAACL 2019 paper (https://aclanthology.org/N19-1162/), neither mentioned in the submission, have explored very similar post-training fine-tuning ideas to improve cross-lingual sentence/contextualized word representations.
All the significant components are adapted from existing work.
Considering the above, combined with the significant improvement in results, a better fit for this work would be the industrial/demo track of a conference rather than a main conference that aims at more scientific contributions.

- Massive high-quality supervised bitext data is required for training MEXMA.
This restricts possible model generalization, as such data is not always available for low-resource languages.
This also limits the method generalization to multimodal settings, as in such settings, modalities are more naturally complementary instead of being parallel.

- The analysis of the paper could be improved. See my questions for more details.

### Questions
- In L171 equation, it seems a natural alternative is concatenating the detailed token-level embeddings of A and B in both parts of the objective. Do you have specific intuition on why only concatenating the sentence-level embeddings to the counterpart?
- I understand there's limited space to show the language-specific results of Table 1, but it would be very helpful to see the results in the supplementary material.
- Another interesting analysis would be comparing your model and XLM-R as backbones of SimAlign (https://aclanthology.org/2020.findings-emnlp.147/), which considers automatic word-level alignment extraction from pre-trained cross-lingual sentence encoders. The results will provide insights into the effectiveness breakdown of MEXMA.

### Soundness
4

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper presents MEXMA, a novel approach to enhancing cross-lingual sentence encoders (CLSE) by integrating both sentence-level and token-level objectives. Traditional CLSE methods focus solely on sentence-level objectives, which can result in information loss, particularly at the token level, and negatively impact sentence representation. MEXMA addresses this issue by using sentence representations in one language to predict masked tokens in another language, with updates directly influencing both the sentence representation and all tokens within the encoder. This incorporation of token-level objectives significantly enhances the quality of sentence representations, leading to improved performance in bitext mining and various downstream tasks compared to existing pre-trained CLSE models. The study also explores the information encoded in tokens and how they contribute to the construction of sentence representations.

### Strengths
1. This paper proposes a multi-grained training objective for fine-tuning a pre-trained language model to enhance cross-lingual sentence representation. The method is straightforward to reproduce.

2. Experimental results demonstrate that incorporating token-level objectives into the training of cross-lingual sentence encoders (CLSE) significantly improves the quality of sentence representations, surpassing the performance of current state-of-the-art pre-trained CLSE models in bitext mining and other downstream tasks.

### Weaknesses
1. The concept of learning both sentence-level and token-level alignment has been explored in several studies, such as those by Wei et al. (2021) and Fan et al. (2022). The authors should clarify the differences and advantages of their proposed method compared to these previous works.

2. There are several cross-lingual benchmarks, such as XTREME, that could be used for comparison. The authors are encouraged to evaluate their method against other works using these benchmarks and provide a detailed analysis of their proposed approach in relation to other methods concerning cross-lingual representation.




Wei et al., 2021. On learning universal representations across languages

Fan et al., 2022. Sentiment-aware word and sentence level pre-training for sentiment analysis

### Questions
N/A

### Soundness
2

### Presentation
3

### Contribution
2

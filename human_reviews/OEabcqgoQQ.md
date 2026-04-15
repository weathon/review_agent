# Can Information-Theoretic Generalization Bound Explain the Generalization of Pre-trained Language Model?

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6, 3

## Abstract
Although language models exhibit exceptional generalization capabilities in downstream tasks after extensive text pre-training, the underlying causes behind this generalization remain unclear. Existing studies on information-theoretic generalization bounds suggest that the compression of information stored in the weights (IIW) is a crucial factor influencing a model's ability to generalize, with some experiments indicating a correlation between lower IIW and improved generalization. However, it remains uncertain whether IIW is applicable to pre-trained language models.  In this work, we find that using IIW can explain why the pre-trained language models have better generalization compared to non-pre-trained language models. Unfortunately, we also discover that IIW does not consistently reflect the degree of generalization when applying IIW to study the fine-tuning process of pre-trained language models. We revisit existing IIW estimation methods, highlighting their limitations in accurately estimating IIW based on theoretical and empirical evidence. Our findings suggest that current information-theoretic generalization bounds, constrained by the limitations of IIW estimation methodologies, fail to accurately capture the generalisation performance of pre-trained language models.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
**Main contributions of this paper:**

- Information stored in the weights (IIW) can serve as a proxy for generalization error in pre-trained models but not during fine-tuning.
- The main reason is “some degree of error in precision estimation IIW”

Models: BERT-base, RoBERTa-base,ELECTRA-base

### Strengths
This paper empirically demonstrates the findings on encoder-style language models: Information stored in the weights (IIW) can serve as a proxy for generalization error in pre-trained models but not during fine-tuning.

### Weaknesses
1. Limited verification: this paper focuses on pre-trained language models, but they only used encoder-style models. Most commonly used pre-trained models today are decoder-only, and this has not been validated in this paper. This issue also extends to tasks such as language modeling and other sequence-to-sequence tasks.
2. The contribution is marginal. Methods in Sec. 2 are all from previous papers. What is new from you? The theoretical analysis in Sec. 4 is lack of experiments verification. 
3. Although the experimental analysis identified precision error in IIW for improvement, it did not provide any solutions. I believe these insights  can be figured out through extensive experiments, but the most crucial aspect is how to solve this issue, which the paper does not address.

### Questions
See in Weaknesses.

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper explores whether information-theoretic generalization bounds, particularly information stored in the weights (IIW), can explain the generalization capabilities of pretrained language models compared to non-pretrained ones. The authors aim to assess if IIW can serve as a reliable measure for generalization performance, especially during the fine-tuning of these models.

The motivation of this work comes from the remarkable success of pretrained LMs (such as BERT, RoBERTa, and ELECTRA) in various NLP tasks. But, the reasons for their good generalization performance is still unclear. Recent studies in information theory suggest that IIW, which measures the amount of information stored in a model's weights, may be a factor influencing generalization. Models with lower IIW are thought to have better generalization performance due to a smaller upper bound on generalization error. However, it is unknown whether this theory holds true for pre-trained language models. Given that, in this study, the authors investigate two main questions: 1) Can IIW explain why pre-trained language models generalize better than non-pre-trained models? and 2) Can IIW serve as a reliable proxy for generalization performance during the fine-tuning process?

To answer these questions, the authors conduct experiments comparing the IIW and performance of pretrained and non-pretrained LMs across various datasets. They also explore how IIW behaves during fine-tuning and whether it correlates with test performance.

### Strengths
- The exploration of IIW in the context of language models like BERT, RoBERTa, and ELECTRA fills an important gap in understanding the generalization capabilities of these models. The authors contribute by providing empirical evidence that pretrained models have lower IIW compared to non-pretrained models, suggesting that IIW can explain why pretrained models generalize better. This is a significant step forward in applying theoretical concepts to language models.

- The paper goes beyond aggregate IIW values, and provides a layer-specific analysis of IIW for both pretrained and non-pretrained models. This level of granularity helps uncover important insights into how different layers contribute to overall model generalization. For instance, they observe that the fully connected (fc) layer in pretrained models has significantly higher IIW than other layers, which they attribute to its role in adapting to downstream tasks.

- The authors critically assess the limitations of current methods for estimating IIW, arguing that these methods may lack precision and could explain why IIW does not consistently correlate with generalization performance during fine-tuning. The paper opens up new direction for future research aimed at improving these methods.

### Weaknesses
- In my opinion the main weakness of this study is about one of the main findings that shows IIW does not reliably serve as a proxy for generalization performance during fine-tuning. In some cases, lower IIW corresponds to worse test performance. This undermines one of the key goals of the paper -- using IIW as a reliable measure for generalization during fine-tuning. The inconsistency between IIW and test performance suggests that either IIW is not an adequate measure for complex models like pretrained language models or that current methods for calculating it are flawed. This weakens the overall impact of their findings since they cannot offer a practical solution for using IIW as a generalization proxy during fine-tuning.

- The authors explore different fine-tuning methods (e.g., full fine-tuning vs linear probing followed by full fine-tuning) but find that lower IIW does not consistently correlate with better test performance across these methods. The lack of consistent results across fine-tuning methods makes it difficult to draw actionable conclusions for practitioners who might want to use IIW as part of their model evaluation process.

- Without offering practical solutions or next steps for improving IIW estimation, the paper leaves readers with more questions than answers regarding how to apply its findings in real-world scenarios. This limits its usefulness for both researchers and practitioners looking for concrete takeaways.

- The paper mostly focuses on presenting empirical results without clearly testing specific hypotheses about why certain phenomena occur (e.g., why lower IIW does not always correlate with better test performance). A more hypothesis-driven approach could have strengthened the paper by providing clearer explanations or predictions about when and why certain patterns would emerge.

### Questions
- It is not so clear if these findings are generalizable for different architectures, for example for state-of-the-art autoregressive models. If the authors can provide some insights about the generalizability of their approach, that would greatly help to the readers. 

- I think more explanation is needed for the inconsistency between IIW and test performance.

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
3

### Summary
The paper investigates whether Information in Weights (IIW), an information-theoretic metric, can effectively explain the generalization performance of pre-trained language models. The authors first explain the IIW metric from Xu&Raginsky et al.'17 and show that the metric is lower for pre-trained models compared to non pre-trained models. However, IIW does not reliably serve as a generalization proxy during the fine-tuning of pre-trained models. The authors study empirical and theoretical insights into precision concerns of the IIW metric, especially during fine-tuning, when the parameters are allowed to grow unbounded.

### Strengths
The strength of  the paper lies in its motivation to measure existing weight metrics to predict generalization guarantees for pre-trained language models. Through careful experiments, the authors show that the IIW metric can distinguish between pre-trained and non pre-trained models. However, the metric can't be used for predicting generalization guarantees after fine-tuning. Through a careful empirical and theoretical analysis, the authors dive into precision issues of the IIW metric. 

In summary, the findings suggest that while IIW helps explain why pre-trained models outperform others, its utility is limited in fine-tuning and more precise methods are needed to serve as a reliable generalization proxy. Overall, this paper stands as an important step towards developing robust generalization metrics.

### Weaknesses
As such, I don't see clear weaknesses with the current work. I have a couple of questions regarding the experimental setup and the conclusion drawn from the observations.

a) **When is IIW computed?** In table 2, when comparing between FT and LP-FT, when is IIW computed for each training method? Is it computed at the end of training for each method? Furthermore, please consider the following questions:
- Will IIW at initialization be a measure that can different between the two methods? How would IIW for LP-FT after LP (first epoch) phase compare to IIW for FT at initialization?
- Can IIW present better correlation for regularized fine-tuning, where the weights are restricted to stay close to the pre-trained values? A plot comparing IIW, regularization strength, and generalization behavior can further strengthen the statement in proposition 1.
- Furthermore, Kumar et al.'22 proposed to use OOD generalization performance to differentiate between FT and LP-FT? Can IIW relate better to OOD generalization?

b) **Increasing IIW with layer depth:** Plots in figure 1 show that IIW of deeper layers are higher for a pre-training model. Is that a generic metric that could be used to select the optimal set of parameters to fine-tune for optimal/efficient training?

c) **Auto-regressive models like GPT-2:** How would the observations change for auto-regressive models like GPT? Specifically, how would the following observations change?
- IIW of the fully connected (fc) layer is significantly higher than that of other layers, because it's randomly initialized for BERT. However, this is not the case for GPT-2.
- Single token v/s multi token in table 1: How would the IIW behavior change with a fine-tuning task that involves multi-token generation at test time?

d) **Computation necessity for the fisher matrix:** Can the authors discuss about how expensive measuring the fisher matrix is (in terms of wall clock time and number of GPUs/CPUs necessary) for a 125M parameter model? 
- How many training samples are necessary to approximate the fisher matrix well for the reported IIW scores? 
- Also, are there approximate versions of fisher matrix (e.g. zeroth order approximations), that the authors could use to compute the IIW metric?

### Questions
Please check my questions in the section above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
A host of results demonstrate that information stored in weights (IIW) provide an upper bound on generalization error. In this work, the authors demonstrate that IIW can be used to explain why pretrained language models generalize better compared to non-pretrained language models. However, they also demonstrate that this metric is not sufficient to disginuish the generalizability between finetuned language models.

### Strengths
- The paper seeks to understand the factors that drive the performance of pretrained language. Given the widespread adoption of language models, this is a timely topic.
- Moreover, the paper addresses several empirical issues with IIW which I think is important for practitioners especially those working in uncertainty quantification.

### Weaknesses
Overall, I feel that many claims in the paper are not well-supported and the scope of the experiments are quite limited. 

- It is hard for me to pin down exactly what the contributions of the paper are. For example, I could see the paper trying to answer any one of the following questions:
    - What drives the performance of pre-trained language models compared to non-pretrained language models?
    - Can IIW generalization bounds characterize the inductive biases of classes of language models or our learning algorithms?
    - How tight are IIW generalization bounds on language models?
    
    It seems that the paper is trying to do all three, but each one is done at a superficial level. 
    
- All of the experiments in the paper focus on encoder-only language models. This makes the scope of the paper quite limited. It makes me question whether or not these results would generalize to decoder-only language models.
- The tasks being explored are only binary (or ternary) classification tasks. What about more complex tasks of interest to the NLP community like summarization or generative benchmarks?
- Much of the analysis being performed is purely qualitative. Even claims that could be made quantitative or statistically analyzed at not (see lines 292-293, lines 284-286).
- The tables in the paper are hard to interpret (Table 1, Table 2). It is unclear what conclusions should be drawn from these results.
- In lines 203-206, lines 210-213 bold claims are being made without any supporting evidence. For example, why does higher IIW correspond to “memorization,” this relationship is nontrivial. Or what does “knowledge acquired” mean here?

### Questions
- The tables in the paper, illustrate the relationship between IIW and test(?) accuracy. However, the theoretical bound presented is about generalization gap. Are the authors assuming that the training accuracy is always some constant like 100%? If this is not the case, then I doubt the validity of the subsequent analyses.
- The plots in the paper (Fig. 1, Fig. 2) measure IIW at a low granularity–every epoch. Why did the authors choose to do this? Could the rising/falling trajectories of IIW discussed in the literature be more apparent if the authors instead probed IIW every $n$ steps?

### Soundness
1

### Presentation
2

### Contribution
2

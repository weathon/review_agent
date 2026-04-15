# Quality Control at Your Fingertips: Quality-Aware Translation Models

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 6

## Abstract
Maximum-a-posteriori (MAP) decoding is the most widely used decoding strategy for neural machine translation (NMT) models. The underlying assumption is that model probability correlates well with human judgment, with better translations being more likely. However, research has shown that this assumption does not always hold, and decoding strategies which directly optimize a utility function, like Minimum Bayes Risk (MBR) or Quality-Aware decoding can significantly improve translation quality over standard MAP decoding.
The main disadvantage of these methods is that they require an additional model to predict the utility, and additional steps during decoding, which makes the entire process computationally demanding. In this paper, we propose to make the NMT models themselves quality-aware by training them to estimate the quality of their own output. During decoding, we can use the model's own quality estimates to guide the generation process and produce the highest-quality translations possible. We demonstrate that the model can self-evaluate its own output during translation, eliminating the need for a separate quality estimation model. Moreover, we show that using this quality signal as a prompt during MAP decoding can significantly improve translation quality. When using the internal quality estimate to prune the hypothesis space during MBR decoding, we can not only further improve translation quality, but also reduce inference speed by two orders of magnitude.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
they propose to make the NMT models themselves quality-aware by training them to estimate the quality of their own output. During decoding, they can use the model’s own quality estimates to guide the generation process and produce the highest-quality translations possible. They demonstrate that the model can self-evaluate its own output during translation, eliminating the need for a separate quality estimation model.

### Strengths
- This paper trains the so-called quality-aware NMT models, which is somewhat novel.

### Weaknesses
- This paper is not well written, I mean, they just train the model with two tasks at the same time, i.e., translation and sentence-level QE. Their model does not have the ability to evaluate the output's quality, which is their argued contribution.
- From the training procedure, I can not get how the model is aware of the translation quality. The method and experiments are not convincing enough.
- The results listed in the experiments are not enough. Why not report the BLEU score? I guess, the BLEU score is not improved as this paper uses the BLEURT-QE as the quality estimation.

### Questions
- See above

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper presents methods for training an NMT model to perform quality estimation along with translation. One benefit of this paradigm is that the MBR decoding could be performed with a high efficiency. The experiments show that the translation quality could be improved.

### Strengths
The methods are: appending the QE label to the source segment; appending the QE label to the target segment, which are both quite easy to perform.

With the proposed method, the MBR decoding could achieve competitive performance with 10-20 candidates, which is much smaller than reported in previous paper.

### Weaknesses
1. generating a quality label does not necessarily mean that the model has the ability to predict it. I am wondering if there is some disturbances are made to the sentence in the training data, will the proposed model generate the correct quality label (showing the quality goes down)?

2. according to fig.1 , the prediction of quality labels is not good at all. The model seems not to be able to discriminate candidates with different qualities.

3. using QE label as the generation labels seems to be an interesting idea. Will you please give some examples of the same source sentence translated with different QE labels? It would be nice to see the effect demonstrated.

4. I am not quite sure how is the quality difference between two translations with 1 point difference in MetricX or Comet score. It will be better to give some examples to show how the translation quality is improved indeed.

### Questions
See the weakness part for the details.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed two methods to make the NMT model quality aware. One is to prompt the NMT model with a quality score during training, but using the best score during inference time. The other is similar to multi-task learning but in a more unified way by appending the quality score in the target side. Both approaches show promising improvements in translation quality and one of them can work well with the MBR decoding to boost the translation quality further.

### Strengths
The paper targets an interesting and essential problem for NMT, which is both related to the translation safety and quality. It proposes two novel and efficient methods. Both methods are very simple but effective according to the experiments. The paper is also well written and clear to me mostly. I believe the proposed methods have the potential to be applied to large scale translation systems.

### Weaknesses
My concerns are in the questions. If they can be addressed properly, they won't be weakness to me.

### Questions
In conclusion, which one between QA prompting and prediction approaches is your recommendation in the situations including latency sensitive inference and large scale distillation. Please also describe how do you scale your methods in large scale multilingual machine translation system. The experiments highly relies on the model best evaluators. How do you make a cold start on a low resource setting?

On discretizing the quality scores, what if the distribution of the scores are very skew? What issues can you see in this case and how would you resolve them?

Is it possible that the quality score won't be generated during sampling in the prediction approach because it's not guaranteed? How do you handle this situation?

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

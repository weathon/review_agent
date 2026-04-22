# The emergence of the left-right asymmetry in predicting brain activity from LLMs' representations specifically correlates with their formal linguistic competence

- Avg Score: 3.50
- Decision: Reject
- Scores: 2, 2, 6, 4

## Abstract
When humans and large language models (LLMs) process the same text, activations in the LLMs correlate with brain activity measured, e.g., with functional magnetic resonance imaging (fMRI). Moreover, as the training of an LLM progresses, the performance in predicting brain activity from its internal activations improves more in the left hemisphere than in the right one. The aim of the present work is to understand which kind of competence acquired by the LLMs underlies the emergence of this left-right asymmetry. Using the OLMo-2 7B language model at various training checkpoints and fMRI data from English participants, we compare the evolution of the left-right asymmetry in brain scores alongside performance on several benchmarks. We observe that the asymmetry co-emerges with the formal linguistic abilities of the LLM. These abilities are demonstrated in two ways: by the model's capacity to assign a higher probability to an acceptable sentence than to a grammatically unacceptable one within a minimal contrasting pair, or its ability to produce well-formed text. On the opposite, the left-right asymmetry does not correlate with the performance on arithmetic or Dyck language tasks; nor with text-based tasks involving world knowledge and reasoning. We generalize these results to another family of LLMs (Pythia) and another language, namely French. Our observations indicate that the left-right asymmetry in brain predictivity matches the progress in formal linguistic competence (knowledge of linguistic patterns).

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
This paper investigates why left-right hemispheric asymmetry arises when large language model (LLM) activations are used to predict human brain activity during language processing. Using training checkpoints from the OLMo-2 and Pythia model families, the authors track how brain predictivity evolves alongside model competence. They find that left-hemisphere dominance co-emerges with the model's acquisition of formal linguistic skills (e.g., syntax, grammatical acceptability) rather than with non-linguistic or reasoning abilities. The authors clain that this relationship generalizes across models, languages (English and French), and datasets. The results suggest that formal linguistic competence, not world knowledge or reasoning, drives LLM–brain alignment asymmetry.

### Strengths
1. The study offers a cognitive-level explanation for hemispheric asymmetry in brain–LLM alignment, bridging neuroscience and computational linguistics.
2. Tracking asymmetry evolution through training provides rare temporal insight into how representational properties emerge in LLMs.
3. The separation of formal vs. functional linguistic competence provides a theoretically grounded interpretation of model–brain correspondence.

### Weaknesses
1. Authors chose two language acceptability tasks (BLiMP and Zorro). Expts should have been done with more linguistic tasks to claim "formal competence (knowledge of linguistic patterns)" in the abstract. The same holds for functional competence tasks -- just two tasks do not justify a broad claim. For French the comparison is just on one linguistic and one non-linguistic task. Aren't ARC and Hellaswag linguistic tasks where the claim does not seem to hold -- what is the definition of a linguistic task?
2. The study depends on one fMRI dataset where participants were listening to an audiobook. Will these observations and insights hold on other fMRI datasets where participants do other tasks in unclear?
3. Line 168 says "ARC and Hellaswag, the high-level comprehension benchmarks, are not aligned with the left-right brain score asymmetry." From Fig 2, to me it looks like alignment is high for ARC easy task. Is there a metric and a threshold for this alignment based on which these claims of alignment vs not are made?
4. Given Fig 4, Fig 1 is redundant.
5. French expt should have been done with some model trained specifically for French. French results do not seem to be convincing. 
6. It would be nice to do this experimentation to observe patterns across different brain regions rather than just hemisphere level results.

### Questions
1. It would be nice to know x_0 vs \beta fit for the 2 french expts. 
2. You argue that left-right asymmetry reflects formal linguistic competence rather than functional competence. How do you rule out the possibility that this asymmetry instead reflects differences in training data distribution or token-level statistics rather than linguistic structure
3. You report correlations between training progression and asymmetry. Did you test lag effects (e.g., whether changes in linguistic competence precede or follow changes in brain predictivity)?
4. Do you think these results would hold for languages with different lateralization patterns (e.g., logographic or morphologically rich languages)?

### Soundness
3

### Presentation
4

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper investigates the origins and potential causes of the “left-right asymmetry” observed in LM-brain alignment --- i.e., the phenomenon that LM activations are more aligned with brain activity in humans’ left hemispheres than right hemispheres. The authors investigate how this asymmetry evolves across checkpoints of an LM’s training process, evaluating LMs on a variety of linguistic and non-linguistic tasks, in both English and French. The authors find that the left-right asymmetry co-emerges with the LM’s functional linguistic abilities.

### Strengths
The topic of understanding representational alignment across LMs and human brains is interesting and timely.
It is also a plus that the findings are validated in languages beyond English.

### Weaknesses
The focus on distinguishing formal/functional competence could have been motivated more clearly. The question “Is the left-right asymmetry driven more by one type of competence than the other?” (l. 066) comes a bit out of nowhere. If we did find that the asymmetry was driven more by one particular kind of competence, what implications would that have? What would be the theoretical importance of investigating this question, either for neuroscience or AI? Since this question is central to the paper’s experiments and analyses, I found the overall high-level takeaways a bit unclear.
 
Also, acceptability combines many factors, including formal as well as functional language competence. More generally, it would have been useful to cover more tasks beyond syntax for analyzing functional language competence.
 
I was also unsure about some of the experimental design choices. For example, in the text generation experiment, how were the five seed prompts chosen? What temperature did you use to sample outputs from the models? Did you perform any manual validation of the generated sentences? 
 
Finally, I felt that the focus on tracking left-right asymmetry across training time was not clearly motivated. Couldn’t you also evaluate a large set of models and see how asymmetric brain scores correlate with task performance? To be clear, I think the training time analyses are interesting, but I found it a bit unclear what theoretical question they are answering.

### Questions
Below are some questions and more minor suggestions.
 
It was unclear in the text which of the evaluation datasets were novel, and which were taken from previously published work (e.g., BLiMP, Zorro, HellaSwag, ARC). Please add in-text citations for the datasets that are not novel.
 
In Section 2.1, does it make sense to average across voxels of different individuals? Does the spatial normalization process take care of this?
 
Section 2.3 was a bit hard to follow, especially since it is so far ahead of the figures.
 
It’s not accurate to say that the models are making “acceptability judgments on sentences” for the BLiMP and Zorro benchmarks (l. 440). You are comparing the probabilities of strings, not asking models for acceptability judgments, correct?

### Soundness
2

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
The paper tests the brain alignment to left and right brain hemispheres (from the Le Petit Prince fMRI dataset) of LLMs over training (OLMo-2 7B, Pythia 2.8 and 6.9B).
This is inspired by a recent finding that LLMs predict the left hemisphere slightly better (Bonnasse-Gahot & Pallier, 2024).
Differences in models' brain alignment scores across hemispheres are compared to their formal and functional competencies via 7 different benchmarks. The authors claim a strong correspondence during training of brain alignment to formal, but not functional competencies.

### Strengths
1. correspondence between performance (on BLiMP, Zorro, ARC Easy) seems to mirror the left-right brain alignment asymmetry very closely. I have rarely seen task scores and biology measurements mirror each other this closely. The findings are consistent with a recent study by AlKhamissi et al. (2025), although the correspondence of formal competencies with the left-right asymmetry seems to be even stronger than with brain alignment overall.

2. findings are generalized across multiple models.

3. findings are generalized across multiple languages (English, French).

Code available in supplement, and GitHub release promised upon acceptance.

### Weaknesses
My main concern is that the difference in L vs R hemisphere alignment is rather marginal: The overall difference in brain alignment between left and right hemispheres reaches a maximum of 0.02 -- is this a difference that we really think is crucial to investigate? I am genuinely asking, it just seems like a small quantitative phenomena to me but the correspondence with formal competencies is so striking.

From another perspective there is a lack of explanation for why the field should care about this left-right asymmetry. Classic neuroscience studies have claimed a left lateralization of the human language network (e.g., Fedorenko et al.) but with LLMs going well beyond core language processing, it's not obvious to me that we should expect them to be more predictive of the left hemisphere.

It would also have been great to test this on more than one fMRI dataset, but I understand that more models and more datasets would always be nice :)

### Questions
Please help us understand why the left-right asymmetry in models' brain alignment is important.

### Soundness
4

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
2

### Summary
This paper studies why large language models (LLMs) show a left-right hemispheric asymmetry in predicting human brain activity. The main hypothesis is that this asymmetry arises when the LLM gains "formal linguistic competence" (like grammar), not "functional competence" (like reasoning or world knowledge).
The authors analyze how the alignment between LLM representations and fMRI signals changes during training, using the OLMo-2 7B model and both English and French data.

### Strengths
1. Clear focus on a specific neuroscience question.
2. Striking correlation between formal linguistic gains and L-R asymmetry.

### Weaknesses
1. The automatic scoring of generated text depends on another LLM (DeBERTa), raising concerns about model-induced bias.
2. The quantitative analysis uses very few data points, possibly making the conclusions unstable.
3. Averaging fMRI signals across subjects may mask individual differences.

### Questions
1. How do the authors rule out bias in using a model (DeBERTa) to judge another model’s outputs?
2. Why is there no L-R effect for Dyck despite it being a purely formal task?

### Soundness
2

### Presentation
2

### Contribution
2

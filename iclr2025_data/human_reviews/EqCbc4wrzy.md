## Human Reviewer 1

### Summary
The paper presents a new dataset for deception detection and emotion recognition.
The authors evaluate several approaches/features on the task of deception detection, also including emotion information.

### Strengths
The new dataset can be valuable to the community if made publicly available.
The authors present extensive evaluations of existing methods/features on the dataset.

### Weaknesses
The concept of emotion and emotion expression used in the paper is fuzzy.
E.g. what is a "true emotional expression" (line 074) supposed to refer to? One possibility that I might suspect is that the authors refer to whether the emotional expression is aligned with an internal state. This is a highly complex topic as emotions are not displayed directly on e.g. a person's face but are subject to regulation processes and social display rules (e.g. see Schneeberger et al., 2023; Müller et al., 2024).


The method novelty is limited, as the authors only evaluate simple combinations of existing approaches. This is not a very significant issue, as the paper presents itself as a dataset paper, where method novelty is not necessary imo. However, in my understanding it would then be important for an ICLR paper to have a more in-depth analysis of what are the implications of the new dataset for deep learning. We should be able to learn something with it. E.g., what are the remaining challenges, does it tell us something about what the current architectures cannot cover?
On what kinds of examples do the current methods fail?


References are missing for the statement "Most studies on deception detection are designed and evaluated on private datasets, typically with relatively small sample sizes" (line 104).
It would also be helpful to add information on the public availability of the existing datasets in Table 1.

In the description of the emotional experiment (lines 198-207), authors mention that they extended some existing dataset of emotional videos because it only has 6 emotions. In the end there seem to be 8. Which emotions were added and why were they considered to be relevant in the context of deception detection?

Concerning the results - the gains by including personality or emotion into the models are only marginal. Are these gain statistically significant when accounting for randomness (of the ssample/datasets, and of the methods/random number generator intialization)?

Research on deception detection raises a host of ethical issues which are not discussed in the paper.

All in all, the strength of the contribution is not too convincing to me and there are a number of open questions on the motivation/scientific background. Furthermore, the quality of the writing is not good enough at the moment, and the improvements for personality/emotion integration are only marginal.


Misc:
-----

- line 015: "role.Although"
- line 093: "introduced a new multi-modal deception dataset" - the characterisation "new" does not make sense here, as the dataset is already quite old (2015) and of course everything was new when it was introduced.
- line 102: wrongly formatted reference ("Speth Jeremy et al.Speth et al. (2021)"), this occurs several times, e.g. in 126,129 and at many other places.
- line 110: "Subjeet", missing space in "Length(Minutes)"
- line 133: non-grammatical paragraph heading "Individual Difference Deception"
- there are more formal/languageissues throughout the paper. the ones above are just some examples.
- line 376: "egemas"


References:
-----------

Schneeberger, Tanja, et al. "The deep method: Towards computational modeling of the social emotion shame driven by theory, introspection, and social signals." IEEE Transactions on Affective Computing (2023).

Müller, Philipp, et al. "Recognizing Emotion Regulation Strategies from Human Behavior with Large Language Models." arXiv preprint arXiv:2408.04420 (2024).

### Questions
What was the reasoning behind using a movie-based emotion induction procedure in the emotion experiment? Why is this procedure relevant in relation to the deception experiment where emotions are induced in a social situation?

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 2

### Summary
In this study, the authors collected data by conducting extensive interviews to create a multimodal deception dataset featuring 193 participants. This dataset also includes information about the participants' personalities and emotional states. Furthermore, the authors explored how features from different modalities and labels impact and enhance the deception detection task.

### Strengths
The authors conducted the data collection with great effort and provided comprehensive coverage regarding the data collection and label design.

### Weaknesses
1. The 'Introduction' section is somewhat verbose. While the authors did an excellent job of providing detailed background information on deception with examples from various modalities, I believe that some of this content, particularly the details in the second and third paragraphs, would be better suited for the 'Related Work' section.

2. The authors mention "effective incentives" without clarifying what these entail (whether monetary or non-monetary) or how they were distributed. Additionally, there is a typo in Table 1.

3. I suggest that the authors include a statement at the beginning of section 3.3 to clarify the order of the data collection activities. 

4. The abbreviation "DDC" at the end of page 4 is unclear—are you referring to "DCC" instead?

5. The role of the 'interviewer' is crucial in this data collection process, as they are responsible not only for asking questions but also for being deceived and labeling responses. However, the manuscript lacks details about the interviewer's background and responsibilities.

6. More specific information is needed regarding how the anonymity of participants is ensured, including what information has been removed or altered. 

7. The statement, "some videos of the CEVS are outdated and cannot successfully induce corresponding emotions in our pre-experiments," is also unclear. What does "outdated" mean in this context, and why exactly are these videos unable to evoke specific emotions?

8. Additionally, what is the source of the 22 online collected videos, and how did the 12 annotators reach a consensus on the labels?

9. In section 4.4 (feature fusion), the second paragraph is poorly explained. After reading it, I'm still unclear about the structure of the "dedicated emotion recognition model" and what specific emotional cues it utilizes. Descriptions such as "We feed a comprehensive set of emotional expression features into this model, allowing it to learn and adapt to the nuances of emotional communication" are too abstract.

10. Similarly, in the results presented in Table 2 and in the discussions regarding personality and emotional features on pages 7, 8, and 9, there is a lack of detail about what these features are and how they contribute.

11. I would also like more information about the 24 interview questions (as detailed in Appendix A.3) that were carefully designed by experienced psychology researchers. Are there any references or theories they used to create these questions?

Overall, the writing can be significantly improved, as there are many uncertainties and unclear descriptions throughout the manuscript. For example: "Before the interview, details of the emotion scale can be found in Appendix C," and "Some studies confirm that some of the five NEO-FFI (Neuroticism Extraversion-Openness Five-Factor Inventory) dimensions are related to deception." 

In conclusion, it is evident that the work feels rushed, and substantial revisions are necessary to address the missing details. As it stands, it cannot be accepted in its current form.

### Questions
Please see the questions raised under the 'Weaknesses' section.

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
3

### Confidence
5

---

## Human Reviewer 3

### Summary
The authors have introduced a multimodal deception dataset. The main novelty of this dataset is that it includes not only deception features but also individual differences in personality and emotional expression characteristics.

### Strengths
The topic of deception detection is a challenging and important one.

### Weaknesses
My main concerns with this paper are about the novelty, the content and the presentation style. Regarding the content, the contribution of this paper seems quite limited for a top conference on machine learning (alternative and potentially more adequate conferences could be ICMI. CSCW, ACM-MM, etc.). Regarding the presentation, there are several typos and some sections of methods (Section 4.2 for example) are very badly organized and written in a not adequate manner.

### Questions
The 24 questions used by the interviewer with a given subject were formulated specifically for this study?

### Soundness
2

### Presentation
1

### Contribution
1

### Rating
1

### Confidence
5

---

## Human Reviewer 4

### Summary
The manuscript introduces the MDPE dataset, a multimodal resource for contributing to deception detection research. This dataset is organized by more than 104 hours of video, audio, and text recordings from 193 subjects, along with annotations on personality traits and emotional expressions. The authors argued that previous deception detection datasets were limited in scope, often lacking the inclusion of individual characteristics that could enhance deception prediction. The authors also demonstrated the effectiveness of MDPE via a series of experiments, indicating that multimodal and personality-aimed models perform better than unimodal models. The manuscript concludes by emphasizing the potentiality of the dataset for future research with consideration of three key tasks, deception detection, personality recognition, and emotion recognition.

### Strengths
S1: The introduced dataset can be one of the good contributions for deception research, providing diverse modalities with consideration of personality and emotion annotations.
S2: The dataset's size is valuable, compared to other datasets. The authors provide the comprehensive dataset of 193 subjects with more than 104 hours. It represents a variety of demographics, covering a broad range of emotions and incorporating deception and genuine responses.
S3. The authors perform extensive benchmarking with multiple features (e.g., visual, acoustic, textual) and combine them with personality and emotional data, leading to valuable findings about multimodal fusion effectiveness.

### Weaknesses
I think that the first weakness of the manuscript is that the collected dataset includes only Chinese speakers, which are limited in its generalization. As the authors know, how do we apply this approach to future research considering other cultures and languages? Obviously, their selected text module (with Chinese-oriented LLM) is effective for their collected dataset. However, it can be applied to other datasets?

Second, although the authors indicate their approaches, I cannot find any demographic statistics in their collected dataset. Is it well-distributed with consideration of age, gender, and a number of characteristics? 

Third, the authors mentioned that emotional characteristics are included in the dataset, but the manuscript lacks a deep analysis of how these characteristics influence deception, instead focusing more on personality traits. This leads to underutilizing the dataset's emotional potential. Moreover, the authors are required to present whether the emotion recognitions are presented with multi-labels or not.

### Questions
1. How do we adapt MDPE to other languages and cultures? Would personality and emotion still be as relevant in deception detection across different demographics?

2. What kinds of multimodal modules can we consider for future research? As presented in the manuscript, the authors considered several previously presented modules, but there can be additional selections for each module.

3. Although the authors consider multi-task approaches for this topic, I think that it is required to compare the results with other emotion recognition or personality-detection models. How do the authors think?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
3

### Confidence
3
# EXPLEME: A Study in Meme Interpretability, Diving Beyond Input Attribution

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 8, 8

## Abstract
Memes, originally created for humor and social commentary, have evolved into vehicles for offensive and harmful content online. Detecting such content is crucial for upholding the integrity of digital spaces. However, binary classification of memes as offensive or not often falls short in practical applications. Ensuring the reliability of these classifiers and addressing inadvertent biases during training are essential tasks. While numerous input-attribution based interpretability methods exist to shed light on the model's decision-making process, they frequently yield insufficient and semantically irrelevant keywords extracted from input memes. In response, we propose a novel, theoretically grounded approach that extracts meaningful ``tokens" from a global vocabulary, yielding both relevant and exhaustive set of interpretable keywords. This method provides valuable insights into the model's behavior and uncovers hidden meanings within memes, significantly enhancing transparency and fostering user trust. Through comprehensive quantitative and qualitative evaluations, we demonstrate the superior effectiveness of our approach compared to conventional baselines. Our research contributes to a deeper understanding of meme content analysis and the development of more robust and interpretable multimodal systems.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes an explanation approach for memes, which provides a more rich set of relevant keywords with the goal of providing richer background information about the semantics of an input meme. The proposed approach is leveraging multimodal encoding and classification through a Language Model (GPT2) with the goal of generating a set of supporting keywords that are not necessarily appearing in the input image, but are inferred (with the help of GPT2) to be relevant to the input meme. Quantitative and qualitative evaluation provides evidence in favour of the good performance of the proposed approach.

### Strengths
- Motivation and problem formulation are well described.
- Several of the proposed idea elements, e.g. prediction preservation, are quite original and well integrated in the overall framework.
- The experimental results appear promising.

### Weaknesses
- There are certain major methodological issues (cf. Questions).
- The related work coverage is insufficient.
- The quality of the manuscript is below publication standards.

### Questions
The motivation behind this work is clear and very relevant. However, a high-level description/explanation of the inner workings of the proposed method is missing from the introduction.

The literature review of multi-modal hate speech detection is rather poor, comprising all in all 9 lines, with half of them being about text-based detection.

The proposed idea is quite similar to concept bottleneck models; the authors could at least add a reference and discuss the differences between the two approaches.

The quality of writing requires considerable improvement.

The methodology presentation is rather unclear and high-level, which makes it very difficult to understand the details of the implementation and to connect the pieces.

To me there are a couple of methodological flows:
1) The fact that GPT predicts certain keywords (based on the image, text, image-caption features & prediction embeddings) does not serve as an explanation of how the two modalities interact in order to produce hate or not. It just automatically provides the context around the meme. Additionally, based on own experience and experiments with OFA to caption memes, the result is often bad, most of the times unsuccessfully trying to OCR the meme instead of providing a good caption about the background.
2) The biases of GPT model are not alleviated of accounted for in this analysis. It is used as if it was perfect.
3) In general, a set of keywords is hardly sufficient to capture the semantic nuance that most memes carry. In the age of LLMs, one would expect an explainability approach to rely on some more sophisticated generative method to produce explanations that are readable.

The authors claim that the method is applicable to other contexts/tasks but give no evidence of that. Therefore, I feel they should remove this from the list of claimed contributions.

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
The paper discusses the challenge of detecting offensive content in memes, highlighting the shortcomings of binary classification and the need for reliable and unbiased classifiers. The authors propose a novel approach for extracting meaningful and interpretable tokens reflecting the meme’s content, enhancing model transparency and user trust. Their proposed approach constitutes four different stages designed towards the objective of extracting meaningful tokens. The stages include: (a) selecting the most relevant keywords, (b) filtering out data points outside a neighborhood of relevance, (c) ensuring semantic relevance within the candidate keywords, and (d) retaining output-preserving tokens from the candidates as the final set. Their method outperforms conventional interpretability baselines, contributing to meme content analysis and developing interpretable multimodal systems.

### Strengths
1. This paper is well-written and easy to understand. The proposed method is intuitive and built over multiple connected modules. 

2. The approach of using auto-regressive language modeling loss, towards generating classification labels while learning to explicitly induce binary signals as part of the modeling approach is interesting. Although, it's a bit of a question itself as to the reliability of manually mapping these signals to the discrete labels, and then further using them for generative LM-based processing. Hence, it could add to the existing limitations of the approach.

### Weaknesses
1. A rudimentary approach to explainability and contextualization (being discussed from the lens of interpretability) within the context of memes.

2. Although the framework *EXPLEME* extracts implicit but relevant keywords w.r.t. the input meme, this at times might not be sufficient to assist in the process of interpreting the model decision, as it could be viewed as adding additional related keywords, yet without imparting a coherent context to the intended implicit content. 

3. The approach may have some limitations in terms of generalizing for harmfulness categories within the wild (prior distributions are unknown), as the label signals depend on a ‘dix’ (category mapping dictionary). It would be better to have more insights into these aspects.

4. The generalizability of the proposed approach, even though an aspect that can surely be investigated, given the variety of the harmfulness and thematic domains and meme design types, is severely lacking in this work.

5. Interpretability is typically examined while understanding or probing the model decisions (predictions) and breaking down the understanding in terms of the steps or key attributes of the modeling approach (framework) that led to a particular output. This also involves looking at and considering the given input in an as-is manner.
While your work does explore the aspect of understanding the model output in a better way, it does pose the question of whether it is the interpretability or explainability that it primarily intends to address. Even Hase et al., 2020, as cited in the paper, trace the inspiration of LAS to interpretability but build the motivation based on the lack of work towards the ‘simulatability’ of NL explanations, which is a distinct notion from conventional interpretability. 
From what I see it, generating semantically relevant textual cues by leveraging the inherent sampling strategy fundamental to generative LM might not necessarily facilitate model interpretability and could offer some form of explainability in some sense (maybe the term simulatability could be popularized with such paradigms), which has been recently reported in the literature to have made significant strides into, in terms of contextualizing the implicit content conveyed within memes [1] and Desai et al., 2021 and Sharma et al., 2022, as has also been cited in the current work’s literature, essentially questioning the research-gap established.

6. With noteworthy strides demonstrated by multimodal LLMs (MLLMs) like LLaVA [2], miniGPT4 [3], etc., apparently, there is no reference, citation and discussion, w.r.t positioning the proposed solution or comparison with it. As multimodal LLMs have demonstrated remarkable capacity towards not only describing the surface level details pertaining to visual-linguistic grounding but also subtle nuances (reasoned via the LLM attached) conveyed/implied within multimodal content, it becomes imperative to factor in their role while investigating solutions towards interpreting or contextualizing meme-like multimodal content.

7. The direct limitation of not examining or leveraging the latest capabilities of MLLMs is also observed in Table 3, third and fourth examples, which are classic examples of nuances involved in typical memes, which the proposed approach fails to resolve.

[1] MEMEX: Detecting Explanatory Evidence for Memes via Knowledge-Enriched Contextualization (Sharma et al., ACL 2023)   
[2] https://llava-vl.github.io/
[3] https://minigpt-4.github.io/

### Questions
**Q1** *Introduction, third para, third sentence*: The text claims that the keywords linked to model predictions often don't semantically match the input meme. It's unclear if this assertion is based on empirical evidence or previous literature, and a citation for support is recommended. Additionally, while the importance of examining methods to contextualize textually expressed harm is highlighted, what is the take of authors on the consideration of implicit visual cues in memes and their integration into the proposed model framework?


**Q2** Is there anything else, in addition to contextualizing memetic phenomena under consideration, that your work is studying on a broader level? As taking a quick look at the outset of Desai et al., 2021 and Sharma et al., 2022, and their main objectives, they do seem to address multimodal contextualization for memes due to obscured meaning. So stating that “existing methods cannot fully explain the model behavior that is not directly related to the input but has some hidden meaning” might not help in positioning your attempts towards a possible gap that previous works seem to have touched upon. 


**Q3** *Methodology*, 
(a) first line: “The proposed systems combine” → Are there multiple “systems” that you’re proposing? Or your proposed approach/methodology/”system” combines a multimodal encoder and a language model. This is in line with the phrase “Our system follows a two-step strategy…” in the very next para. Please clarify and streamline.

(b) last line: “The incorporation of LM enables us to retrieve the set of explainable out-context keywords that are helpful in interpreting the system and its outcome.” → How do you position your goals and attempts against the recent developments within the field involving multimodal LLMs? How does your proposal compete/position/add on to what such multimodal LLMs can achieve?


**Q4** *Dimensions of the multimodal encoding:* How is it that you are working with the dimensions m X 1 and n X 1 for f_{t} and i_{t}, respectively, when one of the standard variants of pre-trained CLIP model: “openai/clip-vit-base-patch32”, renders a common dimension of 512, representing the joint multimodal representational space? Moreover, even if you are working with m and n as the first dimension sizes of these features, how come both U and V (trainable weight matrices) have x X ko as dimensions when your features represent different sizes? Kindly resolve the ambiguity.

**Q5** *Classifying via LM:* 
(a) “li = argmax(FFN(Mt),dim = 1)”: What is the motivation behind employing an argmax, followed by explicit mapping of the signals to either Offensive vs normal, as against learning representations that could directly be used to condition the LM (GPT2) output? The question is additionally motivated by the general idea that performing argmax operation over FFN output is not typically recommended, as the model is usually trained w.r.t. a smooth loss function and doesn’t necessarily perform hard classification. Did you examine your intermediate signal outputs? Any empirical insights probing this aspect would shed some more light on the questionable reliability of this approach. 

(b) “lab = SumPool(gl ◦ E[dix[l]])”: Does the small circle operator bw the gl and E terms represent element-wise multiplication? In either case, what is the motivation behind implementing interaction between gl output and E terms, when there’s no jointly learnt connection between the two? The effect intended to be captured via this operation isn’t super clear.

**Q6** *Sec 3.2, point about Semantic Relevance,* 
(a) “the meme is encoded using the CLIP Vision encoder”: So the dot product computed bw CLIP text encoding of the keywords from the second step and CLIP visual embedding of the meme image doesn’t factor in the CLIP textual encoding of the meme text? Wouldn’t this lead to a relatively lossy multimodal embedding towards examining the semantic relevance?
(b) “First, we use the trained LM in inference mode”: Is it the LM trained as part of your experiments or the standardized (pre-trained/fine-tuned) one? This isn’t completely clear, and it could have implications on the type of output to expect.

**Q7** *Sec. 3.2*, 
(a) “If the model predicts the same class as it predicted before”: As per my understanding, the previous prediction being referred to here in the LM’s primary output is to be considered as predicted (generated) label, thereby not suggesting any grounding w.r.t. the ground truth label. Can prediction flip for some scenarios suggest rectification of the previously incorrectly predicted (generated) label? Need more clarity here.

(b) “token does not have enough importance”: Do you observe any effect due to missing modality here? As when you reinforce your gen-LM’s output as knowledge bite in text-only form, there could be potential meme scenarios, where you end-up losing key information in terms of the intended subsequent LM’s conditioning.

**Q8** *Sec. 3.3, Alignment vs Optimization tradeoff,* 
(a) “In practical applications, this serves as a filtering mechanism to retain tokens relevant to regions”: Is this an empirically verified finding or an established fact, in which case proper citations are a must?

(b) “We term this phenomenon the ‘Alignment vs. Optimization Trade Off Criteria’”: How does your proposed approach, in its scope, offer functionality or results in ways any different from the conventional implications posed by the standard “Alignment vs Optimization tradeoff”?  

**Q9** *Sec 4.2.1, Comparison with the baselines:* I might have missed it, but what are the explanations being used for the explanation-based evaluation (F1 w/ exp) of the random/baselines? I presume the ones obtained (extracted) using the proposed approach are used for evaluating proposed (and variants). Now whatever may be the case for baselines, is it due to the ineffectiveness of the explanation derivation mechanism (hence the generated explanations) or of the interpretability model baselines themselves?  


**Q10** As part of the results (Table 1), it is interesting to observe a consistent range of the diversity scores (inter/intra) and other metrics as well for e-ball enabled vs disabled scenarios, with top K enabled. How would the authors justify the relevance of the e-ball constraint as part of the proposed approach, with such consistent and barely distinct reproducibility with and without it?

**Q11** *Analysis, Does epsilon ball constraint…, page 7,* 
(a) First line, “Without any ε constraint, we obtain a negative LAS score along with a low ‘comprehensiveness’ score.”: Table 1 suggests more on this, with higher ( and non-negative) LAS and comprehensiveness scores, without e-constraint cases (but with top-K enabled). Please resolve the ambiguity, and clarify the confusion.

(b) Last three lines: The optimal value of epsilon observed (0.01<0.05), suggests a smaller neighborhood, as an ideal scenario for better quality keyword extraction. On the contrary, the corresponding theoretical justification stated for it implies near orthogonality b/w e and delta_{m}f(m) components. Do the two have any direct connection in this scenario that I missed? 


**Q12** *Fig. 2 and Analysis point # iv* “What is the similarity of the retrieved…”, page 8, third line: Firstly specifying that [-E,+E] represents E-neighbourhood, then suggesting Jaccard Similarity spikes at [-0.01, +0.01], renders reading Fig. 2 slightly difficult, as your y-axis is JS, and your x-axis has range [1,9], with x-axis label as E-neighbourhood. How to map the range of [-0.01, +0.01] on either of the axes with these configurations?

**Q13** Table 3, example 3 (91768), CLIP-only stage derives “jew, jew, jew, jew”, and ends up misclassifying a normal meme as an offensive one. Since your proposed approach constitutes several filtering stages, would considering some additional steps like post-processing by deduplicating the keywords generated (like in this case), would have facilitated the required diversity for it to be correctly classified? 


**Q14** *Sec 4.3, last line:* Could a directly learnt input representation for GPT2 have given better results, as compared to an explicit transformation based categorical signal? Was the alternative explored as part of the investigation?


**Q15** Use Case: EXPLEME is designed to filter keywords that are directly linked with appeared entities in the meme. But if the same entity is used in another context, then will EXPLEME be able to perform well? For example, a meme contains an image of Donald Trump’s angry face towards the Mexico border, and simultaneously, the other side of the meme contains the image of Hitler’s smirking face with the text ‘Need suggestions !’. Although Hitler had no direct connection with Mexico and its border-related issues, when the smirk of Hitler is merged with the angry face of Donald Trump, it produces an implicit hate meme. Will EXPLEME be able to generate related keywords that connect both cases?

**Q16** Use Case: An entity such as a celebrity or a politician can be used in both positive and negative roles based on the context. If the entity has a biased association towards the negative connotation (example, Hitler) or a positive one (example, Mahatma Gandhi) but has been used in the opposite connotations, then would EXPLEME be able to suggest insightful interpretation/contextualization?

**Q17** The process of collecting external knowledge texts requires more elaboration. External knowledge snippets are fetched based on predefined tokens. Based on what heuristics are the tokens selected? How do the authors go about augmenting the knowledge with a few tokens that are related to the implicit facts?


**Suggestions/Clarifications**

**S1** *Abstract:* 
(a) “However, binary classification of memes as offensive or not often falls short in practical applications.” → This can be generalized well beyond “offense detection”,  like for other hate speech related aspects/categories as well

(b) ‘“tokens” from a global vocabulary’ → It  might be prudent to characterize the scope of the vocabulary within the context of utilizing a particular LM in such scenarios, rather than suggesting it to represent a global vocabulary, which of course can have different technical implications. 

**S2** Introduction, second para, last line, “...kind of inadvertent...“: “…kind of inadvertent biases…”?

**S3** *Related Work,* 
(a) “when kiela2021hateful introduced a set of benchmarks”: Citation format issue.
(b) first para, last line: “This led to a number of research on de- tecting offensiveness in multimodal media, particularly in memes particularly in memes(Sharma et al., 2020; Kiela et al., 2021; Suryawanshi et al., 2020).” → There could be better citations supporting the argument of the follow-up developments, after Kiela et al., 2021. 

**S4** The first formal mention about the task being addressed and the dataset being utilized is mentioned as part of the Sec 3, Classifying via LM and Sec 4.1, Experimental Setup. It is advisable to mention both at an earlier stage of the write-up to build the necessary backdrop, essential towards a complete understanding of the work and methodology.

**S5** You are addressing the task of offensive vs normal, upon which your entire contextualizing keyword extraction steps depends. It might not be recommended to conflate the concepts of “hate speech” and “offense”, when modeling one phenomena, while working with the dataset, build for the other. Offense is a broader term, and hate speech may involve some form of offense, but converse isn’t always true.  

**S6** Sec 3.2, third last line, “linguistic tokens beyond those originally within”: Rephrase?

**S7** Sec 3.3, Alignment vs Optimization tradeoff, page 4, sixth last line, “gradient-based optimization steps of m”: or “gradient-based optimization steps WITH RESPECT TO m”?

**S8** You invest reasonable effort towards theoretically establishing and then analyzing alignment-related aspects of the trade-off while not so much on the optimisation front. Any relevant presumptions I missed in the write-up?

**S9** Sec 4.2.1, Comparison with the baselines, fifth line, 
(a) “proposed approach by resorting to better obtaining scores”: Check the usage of the word resorting in this statement.

(b) “model in the next 10 rows,”: Table 1 shows 11 rows with similar configurations after the top section involving comparative baseline systems. The table can surely use better segregation and markers highlighting the proposed approach and its variants to avoid any confusion.

**S10** Fig. 3, Caption, various TopK and E values: Is the intention to probe various E values or simply compare E-constraint enable vs disable with the optimally set value of E? Kindly clarify.

**S11** An ablation probing the efficacy of the proposed model without the maximal and semantic relevance stages and only with epsilon-constraint with varying values of E would also be interesting.

### Soundness
1 poor

### Presentation
2 fair

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
In this paper, the authors present a novel approach to extracting meaningful tokens from a global vocabulary for effective model interpretability for memes. They demonstrate the effectiveness of their approach on the Facebook Hateful Meme Dataset.

### Strengths
The paper is very well written. The proposed approach seems task-agnostic and would be relevant for model interpretability across meme understanding tasks such as harmfulness detection, offensiveness detection, etc. The authors perform an extensive evaluation to demonstrate the relevance of their approach.

### Weaknesses
It would be interesting to see the variation in the experiments in terms of the language model used (GPT2 currently), dataset for evaluation, tasks, etc. The proposed approach can be demonstrated to work across tasks, datasets, and language models (as claimed in the introduction/contribution).

### Questions
In the human evaluation, were the evaluators aware that they were evaluating a particular method (Ours, ϵ-ball, CLIP, Integrated Gradient)?

Also, I think that a section on the limitations of the proposed approach can be added (would be relevant for future works).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
2: You are willing to defend your assessment, but it is quite likely that you did not understand the central parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The authors present a method to automatically understand the content of a meme, which is a difficult task, because the text alone can be misinterpreted if the image is not taken into account (as the authors demonstrate with an example). Furthermore, the authors derive the mathematics for merging the text and image understanding and give a proof of their theorem. Finally, various experiments are conducted and analysed.

### Strengths
The paper presents maths and prove their theorem. Unfortunately, I wasn't able to fully check the math, and hence, also not able to fully verify and understand the results.

### Weaknesses
The human classification experiments are conducted by the authors of the paper. This should not be the case.

### Questions
I don't have any question.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

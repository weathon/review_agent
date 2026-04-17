# EditLens: Quantifying the Extent of AI Editing in Text

- Decision: Accept (Poster)
- Scores: 6, 6, 2, 4

## Abstract
A significant proportion of queries to large language models ask them to edit user-provided text, rather than generate new text from scratch. While previous work focuses on detecting fully AI-generated text, we demonstrate that AI-edited text is distinguishable from human-written and AI-generated text. First, we propose using lightweight similarity metrics to quantify the magnitude of AI editing present in a text given the original human-written text and validate these metrics with human annotators. Using these similarity metrics as intermediate supervision, we then train EditLens, a regression model that predicts the amount of AI editing present within a text. Our model achieves state-of-the-art performance on both binary (F1=94.7%) and ternary (F1=90.4%) classification tasks in distinguishing human, AI, and mixed writing. Not only do we show that AI-edited text can be detected, but also that the degree of change made by AI to human writing can be detected, which has implications for authorship attribution, education, and policy. Finally, as a case study, we use our model to analyze the effects of AI-edits applied by Grammarly, a popular writing assistance tool. To encourage further research, we commit to publicly releasing our models and dataset.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper introduces EditLens, a detector that estimates the degree of AI editing in a text under homogeneous mixed authorship, where human and AI contributions are entangled at the token level—rather than simply classifying text as human- or AI-written. The task is formalized as regressing a change-magnitude score from the edited text alone, without access to the original. Using cosine distance of sentence embeddings and a "soft n-grams" precision metric—validated against human judgments—as supervision, EditLens is trained on a synthetic dataset of human texts and their AI-edited variants generated via diverse prompts and LLMs. The model fine-tunes open LLM backbones with QLoRA using a multi-task objective. EditLens achieves strong performance as a classifier and produces monotonic scores with increasing edit intensity. It generalizes well to unseen LLMs and domains, and a Grammarly case study shows instruction-specific edit distributions. Models and data will be released.

### Strengths
- Strong and timely motivation. The paper identifies a real and underexplored gap in AI text detection, i.e., the prevalence of heterogeneous, mixed authorship rather than purely human or AI-generated content. By shifting from binary classification to a continuous estimation of edit intensity, it tackles a far more realistic and societally relevant problem.
- The curated dataset of human texts paired with systematically AI-edited variants offers a rich benchmark for studying varying degrees of human-AI co-authorship. Its structure provides a foundation for future research.
- EditLens demonstrates monotonic response to increasing AI involvement, strong macro-F1 across binary and ternary settings, and clear ablation studies.
- OOD tests and the Grammarly case study provide useful insights into how edit types and prompt categories affect the model's judgments.

### Weaknesses
- The dataset is generated exclusively via AI editing human-written text, capturing only one direction of authorship interaction. Yet heterogeneous mixed authorship fundamentally refers to cases where both humans and AI make substantial contributions to the same text. In many real-world workflows, such as researchers revising AI-generated drafts, the reverse direction (human editing AI text) is equally common and important.
- Limited human study scope and unclear methodology. Only three annotators assessed 100 samples using pairwise/tie protocols, without inter-annotator error analysis. Moreover, the paper does not clearly describe the annotation process, such as annotator recruitment or instructions, which limits the reproducibility and interpretability of the results.
- The supervision target for EditLens is derived from embedding and n-gram similarity metrics, which serve only as proxies for the true extent of AI editing. As these same metrics are also used for evaluation, the model may appear to perform well simply because it learns to reproduce the proxy rather than measure real editing strength. A clearer human-grounded calibration or validation study with more annotators would make the construct more reliable and interpretable.
- While the paper is generally clear, some cross-references are vague and hinder readability. For instance, in about line 206, the authors write "see the Appendix", but do not specify which appendix section actually contains it. Such vague pointers make it difficult for readers to locate the intended material, especially given the length of the supplementary text.

### Questions
- Beyond pairwise "which is more AI-edited," can you provide an absolute magnitude annotation (e.g., 5- or 10-point scale) with more annotators, report inter-rater reliability, and correlate EditLens with that scale?
- How does EditLens fare under paraphrasers/back-translation/"humanizers" and multi-pass mixed human↔AI edits beyond five steps? Any detection degradation curves?
- Since the supervision labels rely on soft n-gram precision and embedding similarity, how sensitive are the results to different choices of parameters (a,b,τ) or sentence encoders? A brief ablation or robustness analysis would help confirm that the reported improvements are not specific to one proxy configuration.
- Does EditLens generalize to the reverse case where humans edit AI-written drafts? If not, do the authors anticipate differences in the learned signal or dataset construction for such cases?

If the authors can substantively address the questions and weaknesses outlined above, I would be inclined to raise my score.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a method to predict the extent to which text has been edited by AI. The method is given a piece of text, and must predict (without seein the original unedited source) the extent to which the text was generated by an LLM. This is possible due to the statistical distinctiveness of AI-text compared to human-written text, and is done by training a regression head on top of a transformer, to regress more embedding/n-gram based metrics, which are shown to correlate with expert opinion of AI-editing.
Experimental results seem very strong, though they are presented in such a way that lacks clarity and makes the paper much less valuable in terms of scientific findings, which is unfortunate. But in short, the model can be used in both binary and ternary classification, and is shown to be robust to other kinds of editings that are not included in the training, giving a sense of generalization.

### Strengths
- AI detection is an important topic, and the framing is novel and interesting, for example the introduced concepts of heterogeneous and homogeneous edited text, which expand prior simplistic views in the literature about text editing.
- The method and use of data is quite clever and validated through an expert annotation.
- The findings indicate that this approach is very strong, as it not only outperforms at the correlation task, but also at the more boxed "classification" tasks, through the selection of a threshold. It is an elegant finding that "relaxing" the problem helps improving performance on the initial problem.

### Weaknesses
Though I am very positive about the work overall, my score is only slightly positive for now. I would be willing to raise my score if the authors commit to improvements in the presentation of the work.
A main limitation of the current submission is the extremely rushed description of results. To name only a few things: (1) APT-Eval is never explained or cited (?) leaving readers unclear of what it is and where it comes from, (2) in 4.3 you mention a "calibration procedure above" that is I believe never defined, (3) Table 2 is never described (?), (4) most results are described in a way that is descriptive but not interpretative: you need to help the reader understand the result and why it is important, not just what it is.
This generally really lowers the value of the submission as is, as a reader can tell that the method is promising (clearly), but is not clear about when it shines and when it doesn't and what leads to the sucess.

I highly recommend picking a subset of the presented findings and going over them adequately, rather than rushing through all of them at breakneck speed.

2. I would also encourage the authors to reflect on the work and describe more scientifically where the method still lags, or where there is still room for improvement. As is, all results describe the EditLens method as "solving" the task to the best extent, which is wonderful, but leaves the reader not knowing what remains unsolved. As the authors of this work, you have this perspective from your deep involvement in this project, and scientific communication requires you to help put the work in perspective. A good paper should not only report promising scientific findings, but also help guide the field forward from there.

### Questions
1. The two proxy metrics used for training achieve similar (0.66-0.67) correlation with experts, how well do they correlate with each other. Is there a combination of both (say min(A,B), mean(A,B), max(A,B)) that correlates better with experts? Why do you keep both, I presume they each add something semantic to the problem. Can you comment on why both are needed?
2. It seems like the threshold (based on Table 1) of EditLens on the binary tasks are still rather close to the extremes (0.04 and 0.96), when one could have expected that these would be closer to the mid-range? Can you comment on why that is observed? Does this mean that the community assumed that a 4% change to a piece of human-written text is the threshold at which there should be disclosure of AI-editing?
3. I wonder to what extent the Prompt CLS helps. Does having a model know about categories of edits help it learn to regress, did you conduct any training ablations?
4. Please see questions listed in the weaknesses. How would you modify the current findings /discussion description to improve the submission's narrative?

### Soundness
4

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces EditLens, a novel approach for the quantification of the magnitude of AI editing present within a given text. To this aim, the authors first propose a set of similarity metrics to quantify the magnitude of AI editing in a text given the original, human-written one. Second, after validating these metrics via human annotators, the EditLens regression model is trained on such scores to predict the amount of AI editing in unseen texts. Experiments were performed using diverse models and datasets, and a case study on the Grammarly writing assistant is presented to further showcase the EditLens' capabilities.

### Strengths
- This paper aims to address an important issue in current AI detection approaches, as simply stating AI vs Human-written is reductive and leads to a lot of false positives in current settings.
- EditLens directly regresses the score corresponding to the degree of AI involvement in a text as a whole, rather than simply returning "human-AI mixed" as in the case of its competing methods.
- Results seems to be promising compared to existing approaches, and the experimental setup covers multiple domain scenarios as well as LLMs.

### Weaknesses
- The proposed approach extensively leverages a set of intermediate supervision metrics to train EditLens. However, these represent a key weakness to me. Specifically, these metrics heavily rely on semantic (dis)similarity among the original human-authored texts and their AI-edited counterparts, being potentially misleading. In this regard, if an edit keeps the same semantic meaning of a text while changing multiple "words", the cosine distance would receive a lighter shift than expected. Note that similar considerations hold for the soft n-grams, since they still account for cosine similarity. Related to the previous point, EditLens is hence optimized to follow a concept of similarity learned by *Linq-Embed-Mistral* (which is in charge of producing embeddings), creating a sort of model-dependent similarity, which might include potential biases due to training data of the encoder, as well as capture stylistic variations rather than actual editing. 
- The human evaluation process is not sufficiently robust, as the authors relied on only three annotators and evaluated merely 100 tasks. Although the reported agreement is moderate, the small number of annotators and limited sample might impact the generalizability of the reported findings, e.g., due to annotator biases or sample variability.
- EditLens is found to achieve strong correlation with edit magnitude metrics, but isn't it by construction? It is trained on edit metric signals, so this correlation should be expected rather than a finding.

### Questions
- Related to the first raised weakness, I wonder if the authors could experiment with similar "attack" scenarios (e.g., similar semantics with tangible edits), to better prove the robustness of the proposed approach. Furthermore, do results change when the encoder is replaced with a similar-performing one? This would better highlight any potential "fitting" of the model biases rather than actual editing.
- I would like to ask the authors how robust EditLens to editing made by humans is. That is, a text that is rephrased by humans, comes with the risk of being flagged as AI edited? Could the authors provide some more insights into false positives?
- Why is EditLens based on soft n-grams found to be better in the binary setting, whereas the cosine backbone performs better in the ternary classification?
- Appendix N has a missing reference to the confusion matrix figure.

### Soundness
2

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
This paper addresses the detection of AI-edited text, moving beyond simple classification to quantify the extent of AI involvement. The authors introduce EDITLENS, a regression model trained to predict a continuous score representing the magnitude of AI editing. To achieve this, they create a large-scale dataset of human texts edited by various LLMs and use lightweight similarity metrics, validated against human judgments, as supervision. Their model achieves state-of-the-art performance when adapted for both binary and ternary classification tasks and demonstrates a nuanced ability to track the intensity of AI polish.

### Strengths
+ The paper addresses a highly relevant and timely problem. The shift from binary AI detection to quantifying the continuous spectrum of AI co-writing is a necessary and practical step forward for the field, with clear applications in areas like academic integrity.

+ The creation and planned release of a large-scale dataset specifically for the task of detecting homogeneous mixed authorship is a significant contribution. This resource will undoubtedly enable further research in this under-explored area.

+ The paper is exceptionally well-written, logically structured, and easy to follow. The figures are effective at illustrating the core concepts and the model's performance.

### Weaknesses
- The paper's fundamental limitation is that EDITLENS is not directly trained to detect the "extent of AI editing," but rather to regress a similarity-driven target (e.g., cosine distance). The authors themselves report only "moderate agreement" (Krippendorff's α ≈ 0.67) between this proxy metric and human judgments. The model is learning to approximate a text similarity function, which may not capture the specific stylistic artifacts and semantic patterns characteristic of AI editing, but rather any significant textual change.

- The study fails to include a crucial control experiment to disentangle the detection of "AI editing" from the detection of "significant editing" in general. The model is exclusively trained on human text edited by AI. A necessary test would be to evaluate its scores on human-written texts that have been substantially edited by humans.

- While the problem formulation is a valuable contribution, the technical novelty of the proposed method is low. The approach consists of fine-tuning a standard language model with a regression head, a well-established technique. The success of EDITLENS is therefore entirely dependent on the quality of the supervision signal, which, as noted in the first point, is a flawed proxy for the actual phenomenon of interest.

### Questions
- Given the moderate agreement between your similarity metrics and human annotators, how can you be confident that the model is learning to detect stylistic cues of AI editing, rather than simply learning to approximate a text similarity function? What specific artifacts of AI editing do you believe the model is capturing that a generic similarity metric would not?

- How would you expect EDITLENS to perform on a text written by a human and then heavily edited by another human (e.g., a student paper revised by a writing tutor)? Would the model assign a high AI-edit score in this case? Answering this seems crucial to understanding if the model is a true "AI editor" detector or a more general "heavy revision" detector.

### Soundness
2

### Presentation
3

### Contribution
2

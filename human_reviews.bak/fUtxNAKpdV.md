# Nougat: Neural Optical Understanding for Academic Documents

- Decision: Accept (poster)
- Scores: 5, 3, 8, 10

## Abstract
Scientific knowledge is predominantly stored in books and scientific journals, often in the form of PDFs. However, the PDF format leads to a loss of semantic information, particularly for mathematical expressions. We propose Nougat (Neural Optical Understanding for Academic Documents), a Visual Transformer model that performs an Optical Character Recognition (OCR) task for processing scientific documents into a markup language, and demonstrate the effectiveness of our model on a new dataset of scientific documents. The proposed approach offers a promising solution to enhance the accessibility of scientific knowledge in the digital age, by bridging the gap between human- readable documents and machine-readable text. We release the models and code to accelerate future work on scientific text recognition.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This article presents an end-to-end model for converting PDF image files into a markup language, with a focus on accurately reconstructing mathematical formulas. The authors also discuss the preprocessing pipeline used to align PDFs with LaTeX source code and provide details on the datasets used. The model can be applied to other types of scanned document images.

### Strengths
1. The proposed model significantly improves the processing of scientific documents into a markup language.

2. The model performs exceptionally well in parsing mathematical formulas, surpassing algorithm-based OCR engines.

3. The method is simple yet effective.

### Weaknesses
1. The lack of novelty in the model architecture’s methodology is a weakness. It suggests that the approach may not bring any new or innovative ideas to the field. It would be better for an industrial track paper.

2. The reported results being limited to English data is a weakness. It is unclear if the model can support other languages, such as Chinese or Japanese. However, it would be interesting to evaluate the transfer learning ability of the model for layout analysis, which could potentially address this limitation.

3. Figure 5 and section 5.4 on repetitions during inference being difficult to follow. It indicates that the explanation or presentation of this aspect of the model may not be clear or well-explained, making it challenging for readers to grasp the concept.

### Questions
1. What does “out-of-domain documents” refer to on page 8, and what is the objective of conducting experiments on repetition data?

2. In Eq. X, does “a” or “b” denote a paragraph index?

3. Why were Table, Plain text, and Math treated as separate modalities for experimentation, and how does this differ from conventional methods?

4. Can the model be directly applied to visually rich document understanding tasks?

5. Would it be better to move the first paragraph in Section 3 Model from page 2 to the related work section?


Other suggestions:  give eq. no and full stops

### Soundness
3 good

### Presentation
2 fair

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper presents a way to build a system that converts an image of a scientific document to its Markup representation from both modeling and data collection perspectives. The use of Markup as its representation enables to handle texts, math equations, tables uniformly by a single model. The model is a basic end-to-end transformer-based autoregressive model. In order to mitigate the common repetition problem of such models, a simple data augmentation method is proposed. It also describes the data construction procedure to create image-Markdown pairs from arXiv where several practical techniques are required. The experimental results show that the transformer-based model is capable for the task and outperforms a baseline system. The code and pre-trained model will be released on GitHub.

### Strengths
I see there are mainly three contributions in the paper:

1. It proposes to use Markdown to extract structure information from images of scientific documents.

The use of such representations (e.g. LaTex, html, etc.) in the OCR domain is not novel (e.g. math equation recognition has been studied actively in the community for many years), but proposing to use such representations for this particular domain and tackling the problem seriously should get some credit. If scientific documents stored as images can be converted to Markdown, there should be great values to humanity.

2. It describes the data construction procedure for the task.

Given the capability of latest ML models, the most challenging part is to find data to train models. Identifying arXiv as a good source for the problem and providing practical procedure to construct a dataset seems to have good values to subsequent studies.

3. It shows a transformer-based end-to-end model works for the task and a simple technique further improves the accuracy.

It is not very surprising that the Donut-based model works for the in-domain data, but it should have a certain value to show that it works well in the setting. The proposed data augmentation is simple but effective, and good analyses are given.

### Weaknesses
I am not very certain about its scientific contributions to the ML community. 

In a sense, this paper narrows down the large problem space of OCR into the limited domain and proposes a specialized solution to it. There is definitely a practical value if we can extract Markdown for scientific documents and it is important to have a technical solution to the problem. However, it is not really surprising that a transformer-based model can do a reasonable job for the task. In my opinion, it is challenging to claim novelty for the adoption of transformer for the task and by releasing code and pre-trained models. In my view, the novelty could be claimed for the problem definition and the data construction procedure. Both may be valuable in a certain field, but I am not very sure how much they are for a top-tier general ML conference. I see a lot of high-quality work in the paper, but I feel ICLR is not the best destination for the paper.

### Questions
I think the important details are well described and there is no particular question.

### Soundness
2 fair

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 3

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a new model (Nougat) for automatic understanding of academic documents.
The authors highlight the main limitations of SOTA methods in the conversion of scientific articles, mainly in dealing with complex objects like tables & mathematical equations. As a result, Nougat addresses end-to-end scientific article understanding with a focus on math equations. The authors develop a large dataset of scientific articles and train Nougat to predict their Markdown formatting. Results show that Nougat outperforms SOTA approaches by a large margin. Examples of article conversion are also presented in the appendix.

### Strengths
- The experimental parameters are meticulously described in the article, with extensive information covering preprocessing, model architecture, and training parameters. Notably, the authors have made both the code and the model available on GitHub, which strengthens the reproducibility of their work. The level of detail in their descriptions provides confidence in the potential to replicate these results.

- The methodology used to create the training dataset is also well described and will be available on GitHub. Scientific articles from three sources (ArXiv, PubMed, and IDL) are used. The process of automatic conversion from LaTeX to Markdown is explained step by step, with particular emphasis on the quality control measures taken. The appendix describes the dataset in detail.

- Nougat outperforms GROBID by a substantial margin in the recognition of plain text, tables, and mathematical expressions. Notably, the significant improvement in handling mathematical equations is particularly impressive, given that this was a significant weakness of GROBID.

### Weaknesses
- Nougat outperforms GROBID by a substantial margin in the recognition of plain text, tables, and mathematical expressions. However, I feel that the results section is lacking a qualitative comparison between GROBID, Nougat-base and Nougat-small, especially on mathematical expressions.

- The GROBID system is not sufficiently described, and authors should comment its performance on plain text (what is LaTeX OCR?)

- Two Nougat models are compared in the paper (base and small). However, the article does not provide a comprehensive comparison of these architectures (performance gain, training and inference times, energy consumption…). Consequently, it leaves a significant question unanswered: whether it’s justifiable to opt for the base model when the small model performs almost as well..

- I found the authors’ perspective on how to deal with repetitions very valuable, as this is a well-recognized challenge when working with transformer models. However, the gain should be measured for the repetition detection module, especially since the threshold is set manually and would be painful to adapt to other models/applications.

Finally, some limitations of this work are highlighted, although I feel that this section could be developed further.

### Questions
* Are the scores presented in % for the editdistance metric? This metric is referred to as CER (Character Error Rate) in the document understanding community. 
* This sentence is not clear, as Fig. 2 shows that LaTeX sources are recompiled. In Overleaf you can match paragraphs in the LaTeX source and their localization in the PDF - can this feature be used to simplify this step?
> Since we are not recompiling the LaTeX sources for each paper, we must heuristically split the source file into parts, which correspond to different pages. To achieve that we are using the embedded text on the PDF page and match it to source text. 
* The label noise strategy is not clear. Is noise added to 10% of labels?
> This process continues until the newly sampled number is greater than a specified threshold (in this case, 10%)
* Text vs equations vs tables all have different semantics. Could you comment on which metrics are the most relevant for each text category? It would be interesting to use specific metrics for equation recognition (see the CHROME competition https://www.cs.rit.edu/~rlaz/files/CROHME+TFD%E2%80%932019.pdf)
* The limitation of 4096 tokens seems low for a full page. Did you encounter pages with a larger number of tokens to predict, and how did you handle them? How would this limitation affect document-level recognition (sliding window/patch?).
* Is any post-processing strategy used to normalize section/subsection formatting (example: `##` (section) predicted instead of `###` (subsection)) ?
* Adaptability to other languages? While most publications are written in English, some of them are written in other languages. How much work to fine-tune Nougat? (since the tokenizer, dataset, pre-trained models are all specialized for English) 
* What if repetitions appear inside the page and not at the end? (ex: a sentence is repeated twice, but after that the model continues reading until the end of the page). In this case your method will not work as it will just stop the generation. 1) does this case happen? 2) how would you deal with this issue? 
* Do you handle references to other parts of the page? (Ref to tables, figures in the text?). It might not be crucial when dealing with pages, but it will be useful at document-level.
* Do you handle references to the original image/document (e.g. select a word in the Markdown to highlight it in the PDF)? This feature could be used by users to check an equation if it does not make any sense, for example.
* To go beyond recognition, how could Nougat be adapted to a multitask setting with e.g. Visual Question Answering / Neural Machine Translation / Summarization? It would certainly be useful to allow users to ask questions about the article: ask for an explanation, ask for a reference, ask to translate or summarize...

Errors/typo
* Table A.1. The total does not add up to the number of pages for each source (should be 8,494,841). As a consequence, % for each source are also wrong.
* Typo p9 (last paragraph of section 5.4) "to compute the to the end "

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 4

### Rating
10: strong accept, should be highlighted at the conference

### Rating Number
10

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes an image-to-LaTeX model for converting PDF documents into the corresponding LaTeX source. The model comprises of Swin Transformer encoder, followed by a standard text decoder (BART). The paper also describes in detail an involved process of collecting the data to obtain the training dataset, which involves non-trivial steps to clean and split PDFs. The model and the code to generate the dataset are made available, and the model outperforms several existing benchmarks.

### Strengths
Significance: The paper has an excellent contribution, first and foremost by creating the dataset for pdf-2-latex and describing the methodology and individual steps behind it - these two will greatly benefit the community. The availability of the model is also a significant contribution.
Originality: The proposed modelling approach is not original, but the construction of the dataset is.
Quality & Clarity: The paper is well written and clear, and reproducibility is further fostered by the model / data generation release.

### Weaknesses
No significant weaknesses.

### Questions
One of the central topics in the results is about the repetition of the same sentence again, an artifact of the greedy sampling, and authors approach this problem by doing a data augmentation to help reduce those repetitions. Is there a particular reason why the authors chose to stick with greedy sampling instead of using Top-K / Top-p sampling, which is a standard approach to reduce the overconfidence during autoregressive decoding? (See ex. "The curious case of neural text degeneration")

### Soundness
3 good

### Presentation
3 good

### Contribution
4 excellent

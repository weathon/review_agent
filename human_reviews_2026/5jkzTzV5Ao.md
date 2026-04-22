# Understanding the Implicit Biases of Design Choices for Time Series Foundation Models

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 4, 6, 6, 6

## Abstract
Time series foundation models (TSFMs) are a potential class of powerful, general-purpose tools for forecasting and related temporal tasks, but their behavior is strongly shaped by subtle inductive biases in their design. 
Rather than developing a new model and claiming that it is better than existing TSFMs, e.g., by winning on existing benchmarks, our objective is to understand how the various "knobs" of the training process affect model quality. 
Using a mix of theory and controlled empirical evaluation, we identify and show how various design choices (e.g., patch size, embedding choice, training objective, etc.) lead to implicit biases in fundamental model properties (e.g., temporal behavior, geometric structure, how aggressively or not the model regresses to the mean, etc.), and how these biases can be intuitive or counterintuitive, depending on properties of the model and data. 
We illustrate in a case study on outlier handling how multiple biases interact in complex ways.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper did a investigation on implicit biases in Time Series Foundation Models (TSFMs). These implicit biases are introduced by several design knobs such as patch size, embedding type, and training loss. Instead of proposing a new model, this paper systematically analyzes how these design choices the model’s internal representations and generalization behavior. In this paper, it mainly identifies and discuss three main biases: temporal bias, geometric bias and regression to the mean bias. Combined theoretical analysis with empirical validation, it explains and shows how each biases cause issues and how to improve them.

### Strengths
1: This paper shows a wealth of visualization, including different type of figures. These figures are very easy to understand.

2: The motivation of this paper is very clear. It shifts the focus from performance chasing to understanding inductive biases, highly relevant for Time Series Foundation Models (TSFMs)'s  robustness and transferability.

3: This paper did a deep analysis on each implicit bias, validating findings across several leading Time Series Foundation Models (TSFMs). It also shows practical insights on each bias's investigation.

### Weaknesses
1: For theorem 1, it seems that its theoretical scope is a little bit narrow and relies on simplified assumptions. In Theorem 1, it assumes a linearized ReLU MLP and random Gaussian projections to derive the property of patch embeddings. However, in modern Time Series Foundation Models (TSFMs), they have some attention and temporal positional encoding to dominate their feature engineering. We didn't see any assumptions or discussions on these. We are not sure if this paper's conclusions can hold for transformers trained end-to-end.

2: When this paper discussed about low-frequency preference to large patch size, it didn't provide enough quantitative analysis of how this emerges during training. For example, visualization of any frequency-domain activations through epochs may support this claim. More explanations and rigorous proofs are needed to clarify these points. 

3: For its geometric bias, its geometry discussion shows that angle between embedded vectors for discrete embeddings is much larger than that for continuous embeddings. Also, this grows with the number of bins. However, this paper didn't measure vocabulary size systematically over this observation shown in its figure. More ablation study should be discussed and provided to show that whether the reported geometry is an inherent property of discretization or an artifact of one particular size.

### Questions
Most of questions are mentioned in weaknesses.

1: Could you please give more evidence to support how large patches can lead to low-frequency alignment? especially any end-to-end training support?

2: Could you include structural evaluation metrics to show your conclusions for model's structural correctness?

3: Have you tested your conclusions or findings on real industrial datasets with missing values, multi-seasonality, and domain shifts? How will your findings help the improvements?

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
3

### Summary
The paper presents a study of implicit biases that follow from the design choices of time-series foundation models. Five biases are categorized into two categories, namely temporal (2) and geometry (3) biases. The biases are subtle but are revealed through experiments and proofs (given in the appendix).

### Strengths
* The topic and analysis are novel. 
* I expect this paper to be highly significant for time-series modelling relying on deep learning in general and not only for research in time-series foundation models.
* The paper is extremely thorough.
* The structure of the paper is nice.
* I believe the conclusion to be stronger than what is stated in the paper (see question section).

### Weaknesses
While the writing generally is good, this paper has some issues that I would like to point out. 
* The different biases are not defined explicitly in text. For example, what is meant by temporal and geometric bases? Please specify.  What is the frequency bias and the periodicity bias? What are the angels, distances and norms biases? Explicitly state this, preferably right after the bold heading for each term.
* For temporal bias, frequency, periodicity and seasonality are mentioned, but while frequency and periodicity have their own bias, seasonality is not mentioned again explicitly. What is the difference between seasonality and periodicity?
* On line 149: “In general, these models show different inductive biases …” Which? Be explicit and give examples. 
* The term frequence seems to be overloaded and sometimes mean frequency as measured by Hz and at other times mean “Fourier modes”. When it means one thing and when it means another thing is not explicit. Thie reduces the readability. I would suggest to not use frequency to indicate “Fourier modes” and use a different term instead?
* For geometry bias, paragraph starting on line 267, it is not clear what geometry bias means, as mentioned above, nor which biases are grouped under this term. This is stated for temporal biases.
* The term “breaking the geometry” is used on line 280, but as it is not clear what is meant by geometry, the term is not easily understood. 
* Line 314: Please explain what is meant by more complex reasoning. 
* Not clear what is meant by “the geometry of the input domain”.
* Section 5 Mixture of Biases is a bit short. It would improve with more thorough explanations. 

In summary, this paper is not very easy to read. However, this should be fairly easy to correct given that more space is available for final submission (and in the rebuttal phase). 

Given the depth of the analysis and the complexity of the topic, it would probably improve by having less restrictions on page numbers. Publishing the paper in a journal could be an alternative to a conference because of the share amount of work that is reported. I realize that few journals have the reputation of ICLR and that publication might take longer. 

There are issues with the PDF. It is heavy. The problem is especially noteable on page 7. I assume this is because of Figures 5 and 7. Is it because the figures contain text? Is the text the cause of the problem? They seem like vector graphics and not high-resolution images when I zoom in. Still, the page loads terribly slowly. This issue should be rectified if accepted. 

Improving the text will not only improve the presentation score, but also the soundness, as it will be easier to evaluate it with better presentation.

### Questions
On line 346, what seems like a very important insight is mentioned, but only implicitly. I read it as: Because of the findings in this paper, foundation models for time-series might not be possible because the different design decisions will affect the performance on different tasks. Therefore, TSFMs are task specific, which is not the case for language models as they all operate on the same type of representation (language) with similar characteristics, as words and sentences. Is this the conclusion, or am I misunderstanding something? If this is the case, then the argument should probably be made explicitly in the conclusion. 

Does the summary of results in Figure 1 that is described in text from line 93 reflect the summaries given in the boxes after the description of the biases? For example, Patch Size $\rightarrow$ Temporal bias. However, patch size is a subset of temporal bias, but it also contains architectural choices and unmasked [REG] tokens as well? Did I understand this correctly?

### Soundness
3

### Presentation
2

### Contribution
4

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper discusses how different architectural choices in modern time series foundation models lead to learning biases in the resulting models. In particular, it focuses on biases in modeling the dynamics of input time series and in the structure of the predictions. Each observation is supported by convincing empirical results and well-designed experiments. I found the paper interesting and believe it is a relevant contribution to our understanding of the properties of current time series forecasting architectures. Although there are some presentation issues, I believe the paper is a solid contribution to the conference.

### Strengths
* The paper studies interesting and important phenomena in time series forecasting models.  
* The findings are particularly relevant in the context of foundation models.  
* I appreciate the focus on clarifying the impact of design choices that are often overlooked.  
* The empirical analysis is interesting and well-designed.

### Weaknesses
### Main weaknesses

The writing and presentation could be improved, and several claims would benefit from additional discussion and supporting evidence.

- The introduction could do a better job summarizing the main takeaways of the paper and explaining why the discussed phenomena are particularly important in the context of foundation models (i.e., when learning a transferable model). Some sentences are difficult to contextualize without having read the full paper — for example:  “that time is continuous and this continuity should be maintained, and that regression algorithms should regress to the mean,” or “while a quantization-based embedding introduces a continuous-to-discrete ‘unrounding’ bias, and then relies on training to (imperfectly) recover continuous information in the hidden space.”  Moreover, the example on Chronos vs. Chronos-Bolt performance on chaotic systems is not self-contained and not particularly informative without the cited paper. I would use the introduction to provide more context and clearer motivations for the study. To clarify, I find the motivations sound — I am only referring to the quality of the presentation.  
- I found Theorem 1 quite difficult to parse, as the terminology and assumptions are packed into a few lines. I understand its purpose, but I recommend streamlining the presentation and expanding the discussion on assumptions and their implications. While the appendix provides detailed explanations (which I appreciate), the main text does not adequately contextualize the theorem or its significance. For instance, the assumptions on the weight and bias matrices seem unreasonable without further discussion.  
- The discussion on periodicity bias — specifically the impact of architecture (encoder vs. decoder vs. encoder-decoder), REG token, and patch size — is rather limited in the main body. I understand the space constraints, but as it stands, some claims appear less convincing without looking at the appendices.  
- In Section 3, please clarify the notation. If I understand correctly, *x* and *y* refer to arbitrary patches, and since *k = 1*, these are scalar. Please make this explicit in the text.  
- “When a period is far from zero, its embedded vectors occupy a larger portion of the embedded context, which makes it easier for the model to learn. [...]”  This is unclear and should be elaborated on further. Please clarify and/or provide an example.  
- Please include the related work section in the main paper (you can make it more compact). Since much of the analysis focuses on comparing Chronos and Chronos-Bolt, a more in-depth comparison of the two architectures should be provided — particularly regarding the tokenization mechanism used in Chronos. I understand the space limitations, but I would recommend moving some empirical results to the appendix (or making the introduction more concise) to make room for a clearer context.  
- While some parts of the paper would benefit from more discussion and detail (preliminaries, related work, etc.), other parts could be streamlined (e.g., Sections 3 and 4).  

### Additional comments

- The paper does not mention plans to release the code, but doing so would be very helpful. I also encourage the authors to include a reproducibility statement, as suggested by the ICLR guidelines.  
- Most empirical results are based on synthetic datasets with no noise (systematic noise, not outliers). How do the authors expect the presence of noise to affect their observations?  

Overall, I liked the paper, and if the authors address the issues above and improve the presentation, I'd be happy to increase my score.

### Minor comments

* What do you mean by “first-order structure” (e.g., line 221)?  
* Rather than “periodicity,” I would suggest using “seasonality,” which is the more common term in the time series literature.  
* In Figure 4a, please clarify exactly what *x* and *y* represent.  
* All models used in the study are pretrained, correct?  
* It would be interesting to include experiments on more varied datasets, both synthetic and real.

### Questions
Please comment on the above weaknesses.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper presents a systematic investigation into the implicit biases of modern Time Series Foundation Models (TSFMs). Rather than proposing a new model, the authors aim to understand how common design choices ("knobs") influence model behavior. They identify three primary classes of biases: 1) Temporal Bias, induced by patch size, which affects how models handle different frequencies and periodicities; 2) Geometric Bias, arising from the choice between continuous and discrete (quantization-based) embeddings, which impacts how the model perceives locality, scale, and offsets; and 3) Regression-to-the-Mean Bias, driven by the training loss function (e.g., L1/L2 vs. Cross-Entropy), which determines a model's behavior under uncertainty. The study uses a mix of theory and controlled empirical evaluations, primarily comparing Chronos and Chronos-Bolt, to demonstrate how these biases manifest and interact, ultimately shaping a model's suitability for different types of time series data (e.g., standard benchmarks vs. chaotic systems).

### Strengths
The paper addresses a critical need in the TSFM literature. By shifting the focus from "what is SOTA" to "why models behave the way they do," it provides lasting insights that will remain relevant even as new models emerge. This is the kind of scientific inquiry that fosters deeper understanding.

The combination of theory, controlled synthetic experiments, and analysis on real data is a major strength. The use of Chronos vs. Chronos-Bolt as a primary case study is an elegant experimental design choice that isolates the effects of specific design decisions.

The paper is exceptionally clear. The three-part framework of Temporal, Geometric, and Regression-to-the-mean biases is intuitive, well-supported, and provides a powerful lens through which to view TSFM design. The authors do an excellent job of explaining not just that a bias exists, but how the specific design choice leads to it.

The paper provides practical takeaways. For example, it explains why a model like Chronos (with quantization and cross-entropy loss) might be better for chaotic systems (due to less regression-to-the-mean and a different geometric bias), while a model like Chronos-Bolt might be more robust for noisy, trend-based series. This helps bridge the gap between theory and practice.

### Weaknesses
While the focus on Chronos/Chronos-Bolt is a strength for control, it is also a potential weakness for the generality of the conclusions. The paper does include other models like TimesFM and Moirai in some experiments (which is great!), but the core narrative and many of the detailed analyses are tightly coupled to the Chronos family. It would strengthen the paper to either include more direct evidence from a wider variety of architectures or to more explicitly frame the conclusions in terms of the design choices themselves (e.g., "models using continuous embeddings and L1 loss...") rather than implying they hold for all TSFMs.

The case study on outlier handling in Section 5 is a good first step toward understanding how biases interact. However, this aspect feels somewhat underexplored compared to the detailed analysis of each individual bias. The real world is messy, and models are always operating under a combination of these effects. A slightly deeper discussion or another case study (e.g., handling non-stationarity) could have made the "mixture of biases" a more central contribution.

The inclusion of Theorem 1 is a strength. However, the theorem relies on standard but strong assumptions (e.g., random Gaussian weights) that are not met in a fully trained network. A brief discussion on the expected qualitative persistence of these results in trained models would be helpful to bridge the gap for practitioners. For instance, do we expect the orthogonality of embeddings for different frequencies to be as clean, or just a general tendency?

### Questions
N/A

### Soundness
3

### Presentation
3

### Contribution
2

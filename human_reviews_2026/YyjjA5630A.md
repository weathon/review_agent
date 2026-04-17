# Noise reduction in BERT NER SLM models for clinical entity extraction in clinical trials

- Decision: Reject
- Scores: 2, 2, 6, 6

## Abstract
Precision is of utmost importance in the realm of clinical entity extraction from clinical notes and reports. Encoder Models fine-tuned for Named Entity Recognition (NER) are an efficient choice for this purpose, as they don't hallucinate. We pre-trained an in-house BERT over clinical data and then fine-tuned it for NER. These models performed well on recall but could not close upon the high precision range, needed for clinical models. To address this challenge, we developed a Noise Removal model that refines the output of NER. The NER model assigns token-level entity tags along with probability scores for each token. Our Noise Removal (NR) model then analyzes these probability sequences and classifies predictions as either weak or strong. A naïve approach might involve filtering predictions based on low probability values; however, this method is unreliable. Owing to the characteristics of the SoftMax function, Transformer based architectures often assign disproportionately high confidence scores even to uncertain or weak predictions, making simple thresholding ineffective. To address this issue, we adopted a supervised modeling strategy in which the NR model leverages advanced features such as the Probability Density Map (PDM). The PDM captures the Semantic-Pull effect observed within Transformer embeddings, an effect that manifests in the probability distributions of NER class predictions across token sequences. This approach enables the model to classify predictions as weak or strong with significantly improved accuracy. With these NR models we were able to reduce False Positives across various clinical NER models by 50\% to 90\%.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The authors propose to train a decision tree over various features as a form of confidence calibration over BERT models. The approach is evaluated on two medical named entity recognition datasets.

### Strengths
* There is a clear need for continued work on confidence calibration.
* The authors compare to some common calibration techniques: temperature scaling (2017) and MC-Dropout (2016)

### Weaknesses
Major:
* The authors do not compare to confidence calibration algorithms newer than 2017, e.g., SoftmaxCorr (2024) in the related work, RelCal (2023) https://link.springer.com/chapter/10.1007/978-3-032-05962-8_14, etc. Given the complexity of the proposed feature set for the proposed decision tree classifier, comparisons to such recent work are essential.
* The evaluation datasets are not well defined. No sizes are given. The main dataset is defined only as "EMR data", with no reference for the dataset, or if the dataset was created internally, with no information about the annotation process (e.g., agreement). The MIMIC-3 dataset has no information about inter-annotator agreement.
* Even after looking at appendix A and appendix C, the exact feature definitions are unclear to me.  Features need to be defined formally, with a brief example to demonstrate each feature.
* The paper spends the far too much space (roughly the first four pages) arguing in different ways that confidence calibration is important, and the presentation of the key contribution doesn't show up until Figure 1 on page 7. That content needs to be dramatically compressed, probably down to a maximum of 2 pages so that the key contribution shows up no later than page 3, and so there is room to address the other weaknesses noted here.

Minor:
* Calling BERT a small language model seems odd, given that it's millions of parameters. Consider using a different term.
* This sentence suggests that there are only two important metrics: "Two basic uncertainty metrics can be defined from SoftMax" Consider either justifying the choice of only those two, or include other well-known metrics like margin sampling (best vs. second best), least confident, etc. https://burrsettles.com/pub/settles.activelearning.pdf
* Equation (9) is incorrect; BERT uses positional embeddings, not positional sinusoidal encodings. Consider abbreviating equations 8-18, which are just repeating the standard transformer definition, with something like Z = Transformer(X).

### Questions
* Why K = 3 for CoNLL? Shouldn't it be larger than that since there are E>1 entity types in CoNLL, so K = 2 * E + 1 > 3?
* Could you explain a bit more the intuition behind ProbabilityDensityMap? What is it trying to achieve and why would we expect that to be a good idea?
* Are there triangles (mentioned in the text) in Figure 1? I don't see any.
* Will the annotated subset of MIMIC-3 be released?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper aims to improve the precision of BERT-based small language models for clinical entity extraction task. A simple way is to directly filter out predictions of low probability. However, due to the limitation of Softmax and noise in the training process, this method cannot effectively filter out wrong predictions. This paper proposes a method to distinguish strong and weak predictions by constructing semantic and statistical features for each token position, and training a interpretable decision tree model to classify predictions. This method is lightweight and achieve consistent improvement compared with other baseline prediction filtering methods on EMR and MIMIC-III data.

### Strengths
1. The proposed method can be operated without any modifications to the original NER method or training process, which makes it practical.

2. The main paper gives a detailed explanation in the related work and problem statement.

### Weaknesses
1. Feature motivation. The paper used a case study containing two example clinical texts to illustrate the feature motivation, which is not a persuasive and reasonable way.

2. Experiments. The paper does not give a comprehensive experiment to prove the model, including more comprehensive comparison methods and metrics. The experiment section could be expanded to provide more details of the datasets used, and the inference time for each method may be included to prove the efficiency of NER + NR. Also, a simple ablation study for feature construction may further improve the solidity of the experiments.

3. Paper writing. The paper does not arrange the length and content of its paragraphs appropriately. For example, the results and conclusion sections are too brief, providing neither a clear description of the data used nor an adequate explanation of the comparative methods.

### Questions
1. What do you think are the advantages and practical values of your method compared with deep learning-based and LLM-based approaches in real-world clinical NER?

2. Should a discussion on the interpretability of the decision tree-based NR model be added to the main paper? Why choose exactly these features for decision tree? Though motivation of feature selection is provided in the appendix, this only provides a general idea such as the model should focus on nearby tokens, but not specific feature design.

3. For the experiment section, how are the hyperparameters of baseline methods selected? Are they tuned on the same training dataset as NER + NR?

4. In Table 2, For the biomarkers element, why other methods have almost no FP drop while NER + NR achieve 88% FP drop? This result seems a bit extreme.

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
3

### Summary
This paper presents an analysis of the overconfidence problem in Medical domain adapted pretrained LM-based NER, and engineered features based on the analysis to add a noise reduction model to reduce the false positive predictions.
Contributions
1. Analysis of overconfidence problem in medical NER
2. a well motivated & inexpensive solution with significant improvement

### Strengths
1. The feature design was well motivated from the probability space analysis
2. The proposed solution works well for reducing false positives

### Weaknesses
presentation could be improved. I had to re-read the analysis section 4.2 and 5.1 several times to understand what the authors were trying to say.
1. eq 21 the notation is a mess -- what does |t_anchor-t| mean? I could roughly infer from appendix but it is confusing as hell
2. "feature used" section was confusing -- is 'cross-product' supposed to mean 'cartesian-product'? I had to read the appendix to clarify how it is actually used 
2. better use of space --  the authors spend almost a whole page writing out equations for transformers, and probably resulted in needing to cut out some content to fit in the page limits. I would suggest some parts can be assumed common knowledge or try to compress the space it takes up with some formatting, so you have more room to clarify e.g. your feature design? 
3. orders -- I see probability density maps & binning in section 4.2 density analysis, but did not seem to find the results for these density analysis? and section 4.3 solution overview, maybe better to move them after the feature motivations? the binning part was confusing to me until I read the appendix to see the feature design
4. figures -- the probability space analysis figure is very small, color choices might be a little too similar so it's hard to match which is which in your text to the figure, and since the points will lie on the 2-simplex, maybe adding that triangle can help us better see where the points are in the space?

### Questions
1. In your embedding space analysis, you claimed there's semantic overlap for TP regions and FP regions, but this is after projecting onto 2D, are we sure there is actually overlap or could there be some separating hyperplane say if we add a dimension?
2. I understand this might be out of scope, but in your embedding space analysis/probability space analysis, did any insights come up re improving recall?

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
2

### Summary
The paper builds a simple add-on to clean up mistakes from a BERT model that extracts medical terms. Instead of retraining the model, it uses a small decision tree to spot and remove uncertain predictions by looking at token probabilities and their context. This cuts false positives while keeping recall high andimproving precision for clinical text extraction.

### Strengths
the paper uses interpretable models (decision tree) and that makes it transparent and suitable for clinical settings. the paper also shows effectiveness across multiple datasets and hence the conlcusions can be taken as good generalization. additionally the results are strong (large reduction in false positives with minimal recall loss).

### Weaknesses
The quantitative analysis of htis study is great. my concern is that there is little insight into how the model behaves on more complex or ambiguous clinical text.

### Questions
1. Do you plan to integrate this framework with larger or newer models (e.g., ClinicalBERT or LLaMA-based models)?

2. The quantitative analysis in this study is strong, but could you provide more insight into how the model behaves on complex or ambiguous clinical text?

### Soundness
3

### Presentation
4

### Contribution
4

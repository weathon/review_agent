# (Be Cautious!) Bio-Foundation Models Are Not Yet Robust to Biological Plausible Perturbations and ML Transformations

- Avg Score: 4.00
- Decision: Reject
- Scores: 0, 6, 4, 6

## Abstract
Biological Foundation Models (Bio-FMs) have demonstrated remarkable success across diverse biomedical domains, enabling advances in drug discovery, protein design, and molecular analysis. However, the robustness of Bio-FMs remains underexplored, particularly in terms of the unique risks and perturbations they may encounter in real-world deployment and how these challenges impact their utility. In this work, we characterize the robustness of Bio-FMs from both biology and machine learning (ML) perspectives, and we observe that Bio-FMs are not yet robust to biological data curation and ML transformations. Specifically, (i) from the biological data curation perspective, we design biologically plausible perturbations that mimic corruptions commonly observed in biological experiments, and assess their impact on Bio-FMs; (ii) from the ML perspective, we probe how data transformations, preprocessing, and embedding affect model performance. We systematically evaluate state-of-the-art Bio-FMs on a spectrum of protein-related downstream tasks, spanning protein design, generation, function prediction, cryo-EM reconstruction, and structure classification, over structure, sequence, and image modalities. Our results reveal that most Bio-FMs are vulnerable to both ML transformations and biological perturbations; however, cryo-EM reconstruction models (e.g., CryoDRGN) exhibit a surprising robustness, which maintains stability even under worst-case adversarial scenarios. Notably, we also find that subtle biological perturbations, which are often imperceptible to current measurement tools, yet induce severe discrepancies in Bio-FM outputs, leading to critical failures. Our work highlights underappreciated vulnerabilities and provides a new perspective for evaluating and improving the trustworthiness of Bio-FMs.

## Human Reviews

## Human Reviewer 1

### Rating
0

### Rating Number
0

### Confidence
3

### Summary
This paper investigates how sensitive Biological Foundation Models are to both ML and biological perturbations during inference time. In other words, their behavior out of distribution. ML perturbations are measured by varying graph radius, noise level, maximum neighbors, etc., while biological perturbations are measured by geometric distortions, Gaussian blurring, PGD, name/element scrambling, etc.

### Strengths
The paper is well written and the study is very thorough.

### Weaknesses
The paper examines two types of perturbations: biological and ML, and evaluates the sensitivity of BFMs to each.  

Biological perturbations refer to data alterations such as Gaussian blurring or element scrambling.  
It is unsurprising that these push the model out of distribution, as such perturbations are not part of its training regime.  
When perturbations resemble the noise assumed by the model or when the algorithm (for example, CryoDRGN) includes optimization or retraining steps, the effects are naturally less severe.  
These observations are well known: BFMs, like any tool, assume correct input, and using them outside those assumptions predictably leads to degraded results.  

ML perturbations involve factors such as incorrect hyperparameters or altered molecular graph connectivity.  
Again, these take the model out of its training distribution and predictably harm performance.  
Prior work has already shown that using the wrong noise scheduler in diffusion models or applying a GNN trained on one graph family to another results in underperformance.  

Overall, the paper draws no new or meaningful conclusions. It highlights expected behavior: misusing a model leads to poor results.

### Questions
Issue with citation on line 154-155 (Zhong)
AF3 is trained to predict protein structures mainly from X-ray data. Do the downstream tasks using these AF3 structures assume the same structural bias as X-ray determined structures, which are usually in their minimal energy state?

### Soundness
1

### Presentation
3

### Contribution
1

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper offers a comprehensive analysis of the capabilities and limitations of a suite of biological foundation models (BioFMs) by measuring their robustness against varying degrees of perturbations to their inputs. Their primary finding -- the overall high sensitivity and lack of robustness of most BioFMs -- is a valuable contribution, especially in the context of applications of deep representation learning to the physical sciences. This finding is supported by a large amount of empirical evidence (over 2000 experiments spanning 11 BioFMs, 7 datasets, and 4 categories of downstream tasks) in the context of different kinds of perturbations of the inputs (ML perturbations and biologically plausible perturbations).

### Strengths
- Originality: this is a standard kind of ML paper that measures robustness (or lackthereof) of various deep learning models, and it is overall pursued in a reasonable manner.
- Quality: the primary qualitative strength of this paper is its comprehensiveness in the wide variety of experiments, tasks, models, and datasets they perform the evaluations upon, as well as the framing of two different forms of perturbations (ML vs Bio) and the lack of robustness of these models in the face of both regimes.
- Clarity: the paper is overall clear in its framing, however it would be difficult for any non-Bio reader to follow the details of their experiments and approach.
- Significance: this is a primary strength of the paper in highlighting the limitations of existing FMs, as well as its determination of which BioFMs are more or less resilient than others in handling these kinds of data drifts (e.g., Cryo-EM reconstruction models like CryoDRGN are fairly robust).

### Weaknesses
1. I believe that the paper is overall strong in its content, with one substantive difficulty. Namely, it exclusively evaluates BioFMs and does not compare to traditional baselines or linear models or standard tools. For example, the authors mention both in the abstract and in lines 206-207: "Tiny biological perturbations [...] may be invisible to standard tools but can catastrophically alter Bio-FM outputs". This claim, which I presume is true, is however not substantiated by evidence in the paper (as well, what are these standard tools exactly is not discussed). I believe that the work would be significantly strengthened to a clear Accept if it were possible, for each of the experiments and figures in question (or at least for a subset), to include how the performance of a standard tool or method would change in these benchmarks, under those perturbations. This would be a majorly helpful baseline and point of reference, as we (the community, readers, and the authors) would then have an improved understanding of under what perturbational conditions and at which points (i.e. at what level of perturbation severity) when BioFMs start to underperform compared to standard tools; e.g., for tasks such as function and property prediction. This would also substantiate the claim that the standard tools / non-FM methods are not in fact vulnerable to such perturbations. 

1. While not a weakness per-se, but an opportunity for strength (although arguably beyond the scope of a benchmarking paper) would of course be to leverage these insights to design an improvement to the bio FM regime (or perhaps a more simple finetuning methodology) that can attempt to address these issues of robustness.

1. The other primary weakness is of secondary importance, and I believe could be addressed by the camera-ready deadline. This is the overall presentation and quality of the figures and layout, which could be substantially improved:
   - Figure 1: is it not more standard to have the abstract above the first figure?
   - Figure 1 and 2: why use comic-sans font for parts of these figures? It appears somewhat unprofessional. 
   - 197-198: the use of the plural "questions" is immediately followed by only one question, consider rephrasing.
   - The section titles (3, 4, 5) are rather long, normative claims, and atypical. The rest of the paper includes claims and arguments, I am not sure if section titles should also repeat those (rather than simply being: "Assessing Robustness" / "Robustness to ML transformations" / "Robustness to Biologically Plausible Transformations"
   - Lack of figure standardization in results. Some figures are PNGs and some are PDFs with highlightable text -- typically it is preferred to have all figures be PDFs with highlightable text. Furthermore, various figures have different fonts, font sizes, and coloring standards.
   - The use of in-text Figures may not be standard formatting for ICLR, and it is overall confusing with Figure 8 being referenced before Figure 7 but both being on separate pages. I think figures should not be in-line and instead should be across the page as is done for Figures 3 and 4.
   - Appendix section E repeats the same sentence 3 times (in the section title, in the 1 sentence paragraph, and in the figure caption)

### Questions
- The figures can be confusing -- why does Figure 6 have different y-axis labels for the different models under question? Is that an error, or are Avg. Recovery Rate and Accuracy only relevant measures of task performance for specific kinds of models?

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This work focuses on analyzing the robustness of Bio-FMs from both biology and machine learning (ML) perspectives. In the paper, the authors systematically evaluate various state-of-the-art Bio-FMs on a spectrum of protein-related downstream tasks, spanning protein design, generation, function prediction, cryo-EM reconstruction, and structure classification, over structure, sequence, and image modalities. The results reveal that most Bio-FMs are vulnerable to both ML transformations and biological perturbations. Even though some of the perturbations are not observable by the biological measuring tools, they can still affect the outputs of Bio-FMs.

### Strengths
1, The challenges and motivation behind this paper are clearly shown and justified.

2. Extensive experiments on various state-of-the-art models and downstream tasks provide sufficient analysis to support the findings on the vulnerability of the Bio-FMs.

3. The paper is well written and organized.

### Weaknesses
1. This paper mainly focuses on the findings that Bio-FMs are vulnerable to biological and machine learning perturbations. However, as machine learning models are trained by data, the inherent limitations of these models are vulnerable to input data and process perturbations. Then what are the methods to monitor and prevent the ineffectiveness of Bio-FMs that should also be included in the paper to provide a more complete analysis? Based on my understanding,  methods such as uncertainty quantification have already been proposed to address these issues. Will these methods work on Bio-FMs?

2. When analyzing the machine learning transformation perturbations, the authors conduct the experiments by changing the parameters of the models. This is not sufficient to show that Bio-FMs are vulnerable to perturbations, as the models may become non-optimal when changing the parameters. And the degradation of models' performance is also reasonable under these circumstances.

### Questions
1. In section 4.2, why use the parameter k in a KNN when constructing the multi-relational GNN as the perturbations? Based on my understanding, these are the model parameters. When you change them, you change the model, and of course, non-optimal parameters will lead to the degradation of model performance.

2. Same as the previous question. Changing the parameters in the Bio-FMs is not a good perturbation method. This shows that the models' performance is sensitive to the parameters. But when the optimal parameters are given, whether the model is still sensitive to various protein structures is what is more important.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper presents the first systematic study of robustness in Biological Foundation Models (Bio-FMs) - a class of large models trained on biological data for applications such as protein design, molecular analysis, and drug discovery. The authors approach robustness from two complementary perspectives: (1) Biological data curation -  introducing biologically plausible perturbations (e.g., noise in structure coordinates, folding errors, experimental corruption) that mimic real-world biological data noise. (2) Machine learning transformations - examining ML-specific perturbations such as changes in data preprocessing, feature embeddings, and model hyperparameters (e.g., graph construction parameters like kNN radius).

Across 2,128 experiments on 11 state-of-the-art Bio-FMs, covering 7 datasets and 4 major downstream task types (e.g., protein design, function prediction, cryo-EM reconstruction), the study finds that most Bio-FMs are not robust to even minor perturbations. Cryo-EM reconstruction models (like CryoDRGN) show notable robustness, maintaining stability even under adversarial attacks. Subtle biological perturbations, often undetectable by current tools, can cause critical failures in predictions. The authors highlight the urgent need for robustness benchmarks and provide a conceptual framework to evaluate the trustworthiness of Bio-FMs.

### Strengths
+ This is the first comprehensive robustness study of Bio-FMs, addressing an important gap in current research - trustworthiness in biological modeling.

+ The paper establishes a clear two-dimensional framework (biological vs. ML robustness), bridging disciplinary perspectives in biology and machine learning.

+ The inclusion of multiple modalities (sequence, structure, image) and a large number of models and datasets enhances generalizability and credibility.

### Weaknesses
- While the paper identifies robustness differences (e.g., CryoDRGN’s stability), it provides limited mechanistic insight into why certain architectures or representations yield better robustness.

- The robustness analysis remains entirely in silico. Validation through wet-lab experiments or real-world deployment tests would enhance credibility and biological relevance.

### Questions
Why do Cryo-EM models (e.g., CryoDRGN) exhibit higher robustness compared to sequence or structure based Bio-FMs? Is it due to architecture, data modality, or inherent signal redundancy in cryo-EM images?

Are the biological perturbations calibrated to realistic experimental noise levels or adversarial magnitudes?

### Soundness
3

### Presentation
3

### Contribution
3

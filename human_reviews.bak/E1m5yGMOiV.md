# KinPFN: Bayesian Approximation of RNA Folding Kinetics using Prior-Data Fitted Networks

- Decision: Accept (Poster)
- Scores: 3, 6, 8, 6

## Abstract
RNA is a dynamic biomolecule crucial for cellular regulation, with its function largely determined by its folding into complex structures, while misfolding can lead to multifaceted biological sequelae. During the folding process, RNA traverses through a series of intermediate structural states, with each transition occurring at variable rates that collectively influence the time required to reach the functional form. Understanding these folding kinetics is vital for predicting RNA behavior and optimizing applications in synthetic biology and drug discovery. While in silico kinetic RNA folding simulators are often computationally intensive and time-consuming, accurate approximations of the folding times can already be very informative to assess the efficiency of the folding process. In this work, we present KinPFN, a novel approach that leverages prior-data fitted networks to directly model the posterior predictive distribution of RNA folding times. By training on synthetic data representing arbitrary prior folding times, KinPFN efficiently approximates the cumulative distribution function of RNA folding times in a single forward pass, given only a few initial folding time examples. Our method offers a modular extension to existing RNA kinetics algorithms, promising significant computational speed-ups orders of magnitude faster, while achieving comparable results. We showcase the effectiveness of KinPFN through extensive evaluations and real-world case studies, demonstrating its potential for RNA folding kinetics analysis, its practical relevance, and generalization to other biological data.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
Biomolecules adopt many conformation in situ and understanding the kinetics of the transitions between these conformations is useful for understanding their biophysical behavior. Expensive simulations can start at one conformation and measure how long it takes to transition to another conformation -- the passage times. To minimize the number of simulations needed to characterize the distribution of first passage times, the authors build a prior on the distribution of first passage times; this in principle allows them to do more efficient inference with fewer measurements.

The authors focus on RNA kinetics and expression data. They build their prior by fitting a language model on sequences of passage times. They demonstrate they fit data better than Dirichlet process and kernel density estimator baselines and have reasonable fits on real data.

### Strengths
The paper is sound, and has extensive experimental validation.

### Weaknesses
See questions.

### Questions
1. Why not train on real passage time data that you can simulate? Why synthetic?

2. The prior described by the synthetic data is so simple it may be easy to just run MCMC. Why use a language model at all?

3. Language models as priors can have some pathologies. How does the model behave as the amount of data becomes large? In figure 11 it looks like even with a lot of data the PFN doesn't converge to the true CDF.

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
This paper studies the RNA folding kinetics modeling problem, which is helpful for understanding RNA behavior and designing RNA. The authors propose to apply a deep learning method based on a prior-data fitted network to quickly estimate the distribution of RNA folding times. Experiments in synthetic datasets and real examples demonstrate the effectiveness of the proposed method.

### Strengths
1.	This paper is the first to use deep learning methods to study RNA folding kinetics modeling, an important problem in RNA biology.
2.	On synthetic datasets, the proposed method demonstrates superior performance compared to traditional approaches such as kernel density estimation and Gaussian mixture models.
3.	The proposed method has an advantage in running speed.
4.	The paper provides comprehensive details on dataset construction and model training, ensuring high reproducibility.

### Weaknesses
1.	The current writing does not facilitate quick comprehension of the research problem for readers from diverse backgrounds. It would be beneficial if the author could include a figure illustrating specific data and formalization when introducing the problem. For instance, depicting the relationship between the RNA folding process and the corresponding change in folding fraction could enhance clarity.
2.	The unique challenges RNA folding kinetics pose are not adequately summarized in the introduction. Additionally, the paper directly employs prior-data fitted networks to model the CDF without additional enhancements. Highlighting the improvements made to address the specific issues in this field would enhance the paper's contribution.
3.	It would be clearer to explicitly state in the introduction or background section whether the paper focuses on RNA's tertiary or secondary structure, and how the folding ratio is calculated.
4.	Compared to the dynamic changes in RNA secondary or tertiary structures, the folding ratio provides very coarse-grained information about RNA folding dynamics, which seems still far from practical applications. The paper needs to further elucidate how this study can contribute to solving RNA biology problems.
5.	In the experimental section, the results from Kinfold are used for validation, but the inherent error of Kinfold needs rigorous demonstration, which diminishes the persuasiveness of the results. Is it possible to use collected or published wet lab data for the evaluation of this problem?
6.	tRNA and rRNA are the most common and numerous types of RNA. It would be better to test a broader and more diverse range of RNA types.
7.	There is a lack of research and discussion on deep learning methods suitable for the data in this problem. The paper only presents the prior-data fitted network for deep learning-based probability density estimation.

### Questions
Please refer to the Weaknesses section for details.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
The authors propose a novel method to approximate RNA first passage times (FPTs) using Prior-data fitted networks (PFNs), which are transformers that return a posterior-predictive distribution subject to simulated draws from a prior dataset. In this case, the prior dastset is created from a synthetic prior. The authors put a great deal of effort into hand-crafting this prior (e.g. using biological prior knowledge) that will then be used to train their PFN. The result is a prior that is able to train a PFN that predicts FPT better than competing methods, such as KDE, GMM, and DP-GMMs. The paper's biological motivation is sound and the achievement would help researchers in this field. However, the paper has room for improvement to be published at ICLR instead of a computational biology journal. Why not make the prior adaptive based on the data? There is opportunity to solve this problem that wouldn't require hand-crafting a new prior, or, make it more general for different datasets that would make it an _excellent_ contribution.

The paper's prose is clear. I do think that some of the tables could be rearranged to give the reader more clear insights into how the method works, which I relay below.

### Strengths
- Concrete biological problem of predicting FPTs that was well-motivated and explained.
- Clarity in explanation of method used.
- Clear advantage on benchmarks compared to other current solutions.
- Dramatic decrease in folding time.
- Demonstrated applicability to problems that also exhibit FPT characteristics.

### Weaknesses
- I would have liked to have seen Table 6 comparing all the different methods in the main paper. Plus, I would have liked to have seen comparison on MAE across methods, including in Table 1. 
- Reliance on mulitmodal Gaussians for the prior. 
- Reliance on only two metrics to measure accuracy of the metric.

### Questions
- Why not show the comparison between the sequence length and FPT in Table 1? I think that might provide more insight as to where KinPFN works better over other methods, unless sequence length isn't an important variable, which it seems to be since it's the subject of study in Figure 3.
- Why not put the legend in Figure 5 in the appendix? I'm not going to be able to copy/paste that to check, anyways.
- Why not use the KS test between CDFs as another comparison? This would help capture maximum discrepancy and add nuance to the experimental analysis.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces a model for predicting RNA folding dynamics called KinPFN. Specifically, this work focuses on computing the time needed for an RNA to fold into a specific structure (i.e., the first passage time). KinPFN uses a prior-data fitted network (PFN) that is calibrated using a sample of synthetic first passage time to predict entire cumulative distribution or RNA structures. The methodology is benchmarked on random and real sequences, and the paper concludes with an illustration of an application to gene expression.

Overall, the methodology is sound, and the results are convincing. The development of RNA folding dynamics prediction tools is timely and only a limited set of programs are currently available. Furthermore, machine learning approaches seem suited to this task, and, to my knowledge, this contribution is the first of its kind. The contributed approach is still a proof-of-concept, but it provides solid ground for further exploration of ML. The manuscript is clear and well written, yet it could benefit of clarifications suggested below.

### Strengths
Overall, it is a nice paper from a bioinformatics perspective. The machine learning component is a bit limited but I believe fits the broad definition of originality and significance through "application to a new domain." The authors basically apply one existing framework (prior-data filtering networks or PFNs) to model RNA folding dynamics. The innovative aspect of this work is thus rather the application to RNA folding. The research is timely, and the gain in speed could lead to interesting RNA design applications. The authors mention it but do not discuss it much. However, even if it is still very early, there are interesting/promising applications at the end of the paper. It is a clean and solid piece of work from which could emerge highly useful applications.

### Weaknesses
•	As far as I understand it, the structural model is based on secondary structures and uses (for simulations) the nearest neighbor (NN) energy model. The application to secondary structure is only partially described in the paper. It is an important limitation, and I am concerned that readers may miss this information. I would suggest updating Fig. 2 (reproduced from Muller et al, 2022) to include more details on the sampling (e.g., using a secondary structure model and kinfold) and eventually on the benchmark too. This is just a suggestion as the author(s) may prefer to include this information at other places of the manuscript (e.g., expand the background section to describe the folding model).
•	A claim of the paper is to drastically accelerate folding simulation. I agree this is a nice feature, but I also wonder to what extent it is currently a major bottleneck (for biologists) or what new applications it will enable. I think the manuscript could benefit from further justifications/motivations. For instance, the usefulness of KinPFN for design applications sounds very promising, and I would have liked more discussion (or experiments?) related to this topic. Moreover, since the models currently use kinfold simulations, it could be limited by the accuracy of the NN energy model, which has limitations on longer RNAs. Maybe KinPFN offers new perspectives to deal with this challenge?
•	It is not clear to me that PFNs perform much better than other options (e.g., KDEs) but at least they seem to be suited to the task. Table 1 compares KinPFN to DP-GMM and KDE models. The performance of KDEs seem very close to KingPFN, and I wonder if the author(s) could provide more context in the main body to discuss the significance of the results (e.g., from supp mat H.2)?
•	Fig. 3a shows a distribution of the performance across RNAs of various lengths. It may be out of the scope of this manuscript, but I would be interested to understand better how other features such as the nucleotide composition, MFE value, number of base pairs or loops, etc. impact the results. Also, how does KinPFN perform on multi-stable RNAs?

### Questions
Is there any further conceptual novelty in this submission that I might be missing?

### Soundness
4

### Presentation
4

### Contribution
3

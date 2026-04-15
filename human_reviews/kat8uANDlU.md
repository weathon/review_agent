# Sampling-guided Heterogeneous Graph Neural Network with Temporal Smoothing for Scalable Longitudinal Data Imputation

- Decision: Reject
- Scores: 5, 6, 6, 5, 6

## Abstract
In this paper, we propose a novel framework, the Sampling-guided Heterogeneous Graph Neural Network ($\text{S\small{HT-GNN}}$), to effectively tackle the challenge of missing data imputation in longitudinal studies. Unlike traditional methods, which often require extensive preprocessing to handle irregular or inconsistent missing data, our approach accommodates arbitrary missing data patterns while maintaining computational efficiency. $\text{S\small{HT-GNN}}$ models both observations and covariates as distinct node types, connecting observation nodes at successive time points through subject-specific longitudinal subnetworks, while covariate-observation interactions are represented by attributed edges within bipartite graphs. By leveraging subject-wise mini-batch sampling and a multi-layer temporal smoothing mechanism, $\text{S\small{HT-GNN}}$ efficiently scales to large datasets, while effectively learning node representations and imputing missing data. Extensive experiments on both synthetic and real-world datasets, including the Alzheimer's Disease Neuroimaging Initiative ($\text{A\small{DNI}}$) dataset, demonstrate that $\text{S\small{HT-GNN}}$ significantly outperforms existing imputation methods, even with high missing data rates (e.g., 80\%). The empirical results highlight $\text{S\small{HT-GNN}}$’s robust imputation capabilities and superior performance, particularly in the context of complex, large-scale longitudinal data.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This manuscript proposes a new framework, SHT-GNN, for longitudinal data imputation based on GNN networks. 
It uses the bipartite graph to model the relationship between the covariates and observations and uses a direct graph to model the sequential relations of observations. 
Modeling the imputation as link prediction in the edge utilizes the graph to aggregate the information for an accurate imputation.
Experiments on both synthetic data and one realistic dataset validate its high performance.

### Strengths
S1: The introduction of the method is thoroughly detailed, with complete and comprehensive mathematical formulas provided for each step.

S2: The motivation behind each step is clearly explained and sufficient.

### Weaknesses
W1: The novelty of the proposed method is limited.

W1.1. The method utilizes a bipartite graph to model the relationship between observations and covariates, addressing the task of imputation through link prediction. While similar approaches have been widely used for tabular imputation, as seen in models like GRAPE and IGRM, it appears that the author has merely adapted these methods for longitudinal data, which is essentially not very different from standard tabular data. GRAPE and IGRM also utilize the bipartite graph with U to denote the instance and V to denote the feature. The method proposed in this paper also follows this pipeline besides a directed graph used for instances to model the temporal relationship.

W1.2. It utilizes sampling methods for training the graph neural network, a widely used technique for GNNs. However, the sampling strategy employed here does not appear to be specifically designed for longitudinal data.

W1.3. A temporal smoothing module is proposed to connect different observations; however, its effectiveness has not been studied through ablation analysis. Although some experimental results are provided in section 5.4, these experimental results are conducted on synthetic datasets, which already imply strong temporal relationships. In real-world datasets, this relationship may not exist. 

=======

W2: The presentations of the work can be enhanced.

W2.1. The figures (Figures 1, 2, 3) are not vector images, causing them to become blurry when zoomed in.

W2.2. In Equation 9, the Jaccard distance (or similarity) is not correctly defined according to the standard formula, so it is inappropriate to call it the “Jaccard distance.” In the Jaccard definition, the denominator should be the union of sets A and B. Referring to it as a “Modified Jaccard Distance” may be more appropriate. You can either correct the formula or explicitly state and justify their modified version of the Jaccard distance.

W2.3: In Equation 6, the left side seems to be $h_{u_{i\prime}}$ rather than $h_{u_{i}}$.

W2.3: There are some typos, Row 223 typo, “observation-wish” -> “observation-wise”; Row 303 typo, “and” inside the formula; Row 416 typo, “dicision tree” -> “decision tree”.

======

W3: Some experiments can be enhanced to better highlight its performance.

W3.1: The simulation process in the generation of the datasets is complex and lacks motivation. As provided in the appendix, a 50-dimensional covariate matrix is constructed for each observation. They use a fixed equation with a lot of fixed numbers (hyperparameters). How these numbers or hyper-parameters are selected and while the usage of this fixed equation needs further explanation or citation.
  
W3.2: Some default settings of baselines are changed. For example, in the hidden dimension of the GRAPE, it is 64 by default in its original setting but the authors set it to 16 in the experiments (Line 701)

W3.3: The authors adopt two sets of parameters for different datasets, which may affect the model's adaptability and performance across varying data distributions.

W3.4: The authors include some imputation methods for tabular data (e.g., GRAPE and IGRM) and time series (e.g., CASTI) but they are not SOTA. Some more recent imputation methods such as ReMasker for tabular data [1] and [2] for time series are proposed. Coud you please why these specific baselines were chosen, or to include comparisons with the more recent methods like in [1] and [2].

[1]: ReMasker: Imputing Tabular Data with Masked Autoencoding

[2]: Mining of Switching Sparse Networks for Missing Value Imputation in Multivariate Time Series

### Questions
Q1: How to apply these tabular data imputation like GRAPE and IGRM for longitudinal data.

Q2: How to apply the time-series data imputation method for multiple observations in longitudinal data.

Q3: How do the state-of-the-art tabular imputation methods and time-series imputation methods perform on the longitudinal data?

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
4

### Summary
The authors propose the Sampling-guided Heterogeneous Graph Neural Network (SHT-GNN) approach that explores GNN structures to learn to impute longitudinal irregular sampled multivariate/modal data. The approach relies on GNNs for linking observations through time exploring a subjects minibatched sampling stategy for scalable inference with edge weights defined in terms of temporal smoothing. The latter defined using the product of an exponential decay in time, overlap in missing pattern and cosine similarity of the associated nodes embeddings. To prevent overfitting the Mean Average Distance Gap is used as regularization during training. The approach is contrasted simple to more advanced approaches for irregular data imputation including some existing GNN based imputation methodologies on a synthetic (based on simulated response variables from a real GLOBEM dataset) and real longitudinal dataset (ADNI) finding superior imputation performance than the compared baselines as well as improved computational efficiency when compared to GRAPE and iGRM.

### Strengths
The paper is well written , easy to follow, and with nice illustrations explaining the approach.

The methodology is sound and the included components including temporal smoothing and regularization for oversmoothing in GNN well motivated.

The approach have merits both in compute and ability to impute exploring temporal and multivariate/multimodal structure of data.

### Weaknesses
Whereas the approach is sound and the experimental comparison reasonable, it is somewhat limited. I.e. two datasets of which one is with simulated responses.

Furthermore, the literature on imputation of time-series data is vast and in this space, the paper covers only some of these works, see for instance:
https://paperswithcode.com/task/imputation

Arguably many of these listed procedures do not handle irregular sampled time-series, however, there are still many relevant methods and datasets for time-series imputation of irregular data that it would be highly interesting and relevant to see the performance of the current methodology against beyond the considered baselines. In particular the authors should consult Table 3 in the following recent survey of imputation of irregular time-series data for methods and datasets:
https://www.sciencedirect.com/science/article/pii/S0925231221003003
as well as the recently published article on imputation of irregular time-series data using autoencoders:
https://dl.acm.org/doi/10.1145/3616855.3635831

In particular, health care data such as MIMIC-III and the Physionet 2012 ICU challenge could here be relevant to consider as previously used and potentially also the other ICU datasets here listed. 

Methodology-wise it would be interesting to compare against the recent autoencoder framework of:
https://dl.acm.org/doi/pdf/10.1145/3616855.3635831
as well as, BRITS, GP-VAE, SAITS also here compared against.

Whereas the approach is sound and I believe with merits the experimentation is currently too limited to fully see how meritable the approach is and its impact upon this very large body of existing literature.


Minor:
Dicision Tree -> Decision Tree

### Questions
Could the authors consider including more datasets, for instance as used previously for imputation of irregular data as referenced above? See in particular datasets and methodologies used here:
https://www.sciencedirect.com/science/article/pii/S0167947317300403
https://www.sciencedirect.com/science/article/pii/S0925231221003003
https://dl.acm.org/doi/10.1145/3616855.3635831
In particular, the second reference list many benchmark data sets (Table 3) as well as existing methodologies here used that are not currently compared against. Given the vast literature addressing imputation of irregular time-series data I think the paper needs to establish results much more extensively in terms of data and compared methods and the study is currently in its experimentation rather limited and in my eyes too limited. This makes it hard to judge the impact and utility of the approach compared to this rather large literature. I therefore strongly encourage the authors to include additional experimentation on well-established irregular temporal imputation datasets also considering additional well established methodologies for imputation in such irregular data (as given in the recent references above).

In summary, I think the authors’ methodology is sound and potentially indeed meritable and worth publication, but at this point I find this too unclear as the experimentation in terms of datasets and compared methodologies is too limited.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
The goal of this paper is to present a scalable imputation method for handling missing data in longitudinal studies. A sampling-guided heterogeneous graph neural network (SHT-GNN) is proposed and it is both scalable and capable of learning effectively from irregular and inconsistent longitudinal observations. The proposed method is evaluated against eight baseline methods on synthetic data generated from real data, and it outperforms all baseline methods.

Missing data is a common issue in longitudinal studies, making this an important problem to address. However, imputation methods for longitudinal covariates have been extensively studied, and a key method by Yao et al. (2005) was overlooked and not included in the comparison study. Although this method was designed for Gaussian data, it performs well for non-Gaussian data in practice. It would be valuable to see how this method compares to SHT-GNN.

The authors claim that their method can accommodate arbitrary missing data. However, this may be an overstatement, as it seems likely that their approach is effective only for data missing completely at random. If the method is indeed applicable to missing-at-random or informatively missing schemes, this should be clarified.

Reference: 

Yao, Müller and Wang (2005). Functional data analysis for sparse longitudinal data.
Journal of the American Statistical Association.

### Strengths
The topic of imputation for longitudinal data is significant, and the proposed solution is appealing.

### Weaknesses
An important method was omitted from the comparison study.

### Questions
1. In what contexts are longitudinal observations inconsistent? Can your method handle noise in longitudinal measurements?

2. Does the temporal smoothing method work for longitudinal data with only a few measurements per subject?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a method to impute data in multivariate longitudinal data.

### Strengths
The paper focuses on a timely topic, dealing with longitudinal health data.  Moreover, they apply it to a real-world dataset of interest, ADNI. The developed a method that, at least in theory, overcomes limitations of existing tools.

### Weaknesses
There are 4-5 pages of methods.  Many steps were taken, each with its own few decisions.  It was quite difficult for me to ascertain which steps were important, and how much each step mattered. The ablation study did not help me much, I was not sure precisely whether this was on the real data or the synthetic data. 

The paper only shows experiments in which the SH-GNN method outperforms everything else, for all possible parameter values of the simulation, and also the real data.  That is a bit fishy.  No method outperforms all other methods in all other scenarios.  Especially one that runs faster and takes less memory.  I understand that there are likely no 'benchmark data' for this kind of problem.  And yet, my interests when I read a paper are to get very clear when I *should* use a method, and when I should not.  I get no real insight into that from this manuscript.  The claim seems to be that I should use it when I have multi-subject longitudinal data, with cross-sectional and cross-temporal missingness.  But, there are certainly conditions on the distribution and missingness under which other things are better, both in theory and practice.  What are the conditions?

I find it a bit weird that decision tree was included, rather than random forest, or some other decision forest based approach, since forests are known to be far superior to trees.

### Questions
1. The method contains many steps, I did not follow them all, to be completely honest, there is a lot of new notation and terminology.  It would be great to simply the exposition of the method, if possible.

2. Of all the fancy things included in the method, including temporal smoothing, subject-wise mini-batch sampling, and concatening lots of stuff, experiments that illustrate which of those steps matters most, and how much, would be much more informative.

3. A simulation setting in which something trivial and straightforward outperforms your method, would be instructive.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper proposes Sampling-guided Heterogeneous Graph Neural Network (SHT-GNN) for Longitudinal data imputation. SHT-GNN treats both observations and covariates as nodes, and connects observations at successive time points, while keep covariate-observation interactions in a bipartite graph. Three key component in this model: subject-wise mini-batch sampling to reduce computation burden, inductive learning from observed edges and temporal smoothing in longitudinal subnetworks. With the embeddings generated with GNN, the authors employs MLPs to get the imputed covariates and predict the response. The loss function also takes consideration of a term Mean Average Distance Gap to prevent over-smoothing of GNN. Extensive experiment results and ablation studies are provided with semi-simulated and real data to demonstrate the effectiveness of the proposed framework.

### Strengths
- The paper is well written and easy to follow. Illustrations make it clear to understand the framework.
- The proposed framework, especially the way to construct the graph is novel.
- Experiments show promising results on the provided framework, especially on the real ADNI dataset, which could have significant impact on biomedical science.

### Weaknesses
- The paper is mainly heuristic on proposing a framework and show experiment results, without theoretical understanding of why it works. Although I do not think this is a deal breaker, it would be nice if the authors could provide more intuitions on why they design certain module the way it is. For example, it is unclear whether the way to get the weight with equation (7) is the best option, are there any alternative ways? How stable it is in practice and how to set $\gamma$ in practice?
- For covariate imputation, it seems to me one important baseline is missing, which is to do feature propagation with the proposed graph, similar to the way label propagation works. A recent work is [1]

[1] Rossi, Emanuele, et al. "On the unreasonable effectiveness of feature propagation in learning on graphs with missing node features." Learning on graphs conference. PMLR, 2022.

### Questions
- Is there any transformer based model for longitudinal data imputation? If there is work with sequence models like RNN and LSTM, I would think there should be transformer based models too. If so, they should also be baselines, and they could potentially mitigate the computation burden by parallel computing
- The response simulation model in Appendix 1 seems strange. Are you just combining different linear/non-linear functions together to make it complex or is there any intuition or real world model to guide the design?
- One suggestion is to change the abbreviation to SH-GNN

### Soundness
3

### Presentation
3

### Contribution
3

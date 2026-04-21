# Gene Regulatory Network Inference in the Presence of Dropouts: a Causal View

- Avg Score: 7.33
- Decision: Accept (oral)
- Scores: 8, 8, 6

## Abstract
Gene regulatory network inference (GRNI) is a challenging problem, particularly owing to the presence of zeros in single-cell RNA sequencing data: some are biological zeros representing no gene expression, while some others are technical zeros arising from the sequencing procedure (aka dropouts), which may bias GRNI by distorting the joint distribution of the measured gene expressions. Existing approaches typically handle dropout error via imputation, which may introduce spurious relations as the true joint distribution is generally unidentifiable. To tackle this issue, we introduce a causal graphical model to characterize the dropout mechanism, namely, Causal Dropout Model. We provide a simple yet effective theoretical result: interestingly, the conditional independence (CI) relations in the data with dropouts, after deleting the samples with zero values (regardless if technical or not) for the conditioned variables, are asymptotically identical to the CI relations in the original data without dropouts. This particular test-wise deletion procedure, in which we perform CI tests on the samples without zeros for the conditioned variables, can be seamlessly integrated with existing structure learning approaches including constraint-based and greedy score-based methods, thus giving rise to a principled framework for GRNI in the presence of dropouts. We further show that the causal dropout model can be validated from data, and many existing statistical models to handle dropouts fit into our model as specific parametric instances. Empirical evaluation on synthetic, curated, and real-world experimental transcriptomic data comprehensively demonstrate the efficacy of our method.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors tackle the task of inferring gene regulatory networks (GRNs) from single-cell RNA sequencing (*scRNA-seq*) data, which is made difficult by the fact that *scRNA-seq* data showcases many zero values, which are either due to technical reasons (dropouts) or biological (no gene expression). This missing data problem can lead to overly-dense graphs being produced by SOTA causal discovery algorithms applied to this task. The authors propose a `Causal Dropout Model` to characterize the dropout mechanism and come up with a simple solution to handle dropouts, namely conditioning on non-zero entries in the conditioning set when performing conditional independence sets. They show that this approach is sound under relatively mild assumptions and performs well on a large number of synthetic, semi-synthetic, and real-world data sets.

### Strengths
The problem tackled is very important, as gene regulatory networks can provide direct insight into the workings of biological mechanisms, and *scRNA-seq* is increasingly available for this task. The authors present their idea in an almost flawless manner, with sufficient attention being given to describing related work, providing relevant examples, and to framing and testing the assumptions made for the `Causal Dropout Model`. Finally, the authors showcase the performance of their approach in an extensive series of experiments, in which they examine different dropout mechanisms, different causal discovery and GRN inference-specific algorithms, on multiple settings and types of data.

### Weaknesses
My only (minor) gripes are that most of the theoretical analysis is deferred to the appendix, which can sometimes lead to questions due to insufficient detail (see below), and that the references are a bit sloppy. 

Miscellaneous comments:
- page 4, Example 4: I think the Bernoulli distributions should be reversed, since there should be a minus sign in the denominator exponential when computing the logistic function.
- page 10, reference typo: * after "Tabula Sapiens Consortium"
- page 11, formatting error: "ALBERTS" should not be capitalized
- page 11, there seems to be no reference to Gao et al. (2022) in the paper.
- page 12, duplicate author: "Paul R. Rosenbaum"

### Questions
1. On page 6, when discussing the connection between BIC score and Fisher-Z test statistics, how do you conclude that "only local consistency is actually needed"? The reference in Nandy et al. (2018) shows this connection, but does not say anything about the assumptions made in the GES paper. Could you elaborate on this point?
2. On page 7, what do you mean by "identifiability upper bound" for identifying the dropout mechanisms? How is this upper bound characterized?
3. Perhaps I missed it, but in Theorem 2, what is the "one particular case" in which Z_i and Z_j are non-adjacent in the underlying GRN.
4. In Figure 4(b), how do you explain that testwise deletion performs better than the oracle? Shouldn't that be the best case scenario?
5. I haven't seen *causal sufficiency* mentioned anywhere, which is an important assumption made when using algorithms like PC and GES. Is it reasonable for this type of data to assume that there are no hidden confounders? In either case, I would say something about this important practical limitation.

### Soundness
3 good

### Presentation
4 excellent

### Contribution
4 excellent

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors propose a method for causal discovery of gene regulatory networks (GRNs), i.e., causal gene-gene relationships. Instead of conducting conditional-independence tests on the raw data without consideration of the dropout patterns of single cell sequencing (e.g., doing PC algorithm on the entire data), they propose conducting the tests only on non-zero conditioning variables, by showing that the conditional independence relations of variables conditioned on variables with non-zero values are the same as for data without dropouts. The authors demonstrate that their method outperforms competing state-of-the-art methods in causal discovery of GRNs on several data sets.

### Strengths
- The method is in my opinion very original and will certainly be of actual use in the field of computational biology.
- The paper is clearly written and easy to follow. As far as I can judge, the authors demonstrate an excellent grasp of the contemporary literature, both in causal discovery as well as in computational biology.
- The method outperforms state-of-the-art methods for GRN inference in several benchmarks.
- The experimental section is convincing, and should be easy to reproduce.

### Weaknesses
I do not have major comments on possible weaknesses.

### Questions
- The authors state "while a zero entry of $X_2$ may be noisy (i.e., may be technical), a non-zero entry of $X_2$ must be accurate, i.e., biological.". As far as I can tell, non-zero values might also be technical due to sequencing/mapping/algorithmic errors, correct? 
- Supposing elevated dropout rates of, say 50%, the method relies on the fact that a conditioning variable can be found which "breaks" dependencies. Is this correct?
- Using Fisher's $z$-test assumes multivariate Gaussianity. Wouldn't a kernel-based independence test be better for the log-normal data?
- Figure 5 is not very readable. I believe a simple table or something similar could improve the presentation.
- The source code could be improved and properly documented (What libraries are required? Which versions of the libraries did you use for validation? How do I run all experiments? Etc etc.).

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The authors proposed a causal graphical model, named causal dropout model, to characterize the dropout mechanism in scRNA-seq data. 
They found that simply ignore the data points in which the conditioned variables have zero values can still lead to consistent estimation of conditional independence (CI) relations with those in the original data.

### Strengths
The task of inferring gene regulatory network is of interest especially in the bioinformatic domain.
The empirical results seem to indicate that the approach can be integrated into existing causal discovery methods to handle dropouts.
Writing and presentation skill is well.

### Weaknesses
For network inference, they can use some evaluation metrics such as ROC curve or PR curve to assess how well their predicted network recovers the true network. 
They should conduct more experiments to show the performance gained by using their causal dropout model.
Several gene network inference methods have been designed to handle missing values in scRNA-seq data. Therefore, as a practical analytical framework, the authors should prioritize the comparison of their model with the most advanced existing network inference methods.

### Questions
For network inference, they can use some evaluation metrics such as ROC curve or PR curve to assess how well their predicted network recovers the true network. 
They should conduct more experiments to show the performance gained by using their causal dropout model.
Several gene network inference methods have been designed to handle missing values in scRNA-seq data. Therefore, as a practical analytical framework, the authors should prioritize the comparison of their model with the most advanced existing network inference methods.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

# Rethinking the generalization of drug target affinity prediction algorithms via similarity aware evaluation

- Decision: Accept (Oral)
- Scores: 8, 8, 6, 8

## Abstract
Drug-target binding affinity prediction is a fundamental task for drug discovery. It has been extensively explored in literature and promising results are reported. However, in this paper, we demonstrate that the results may be misleading and cannot be well generalized to real practice. The core observation is that the canonical randomized split of a test set in conventional evaluation leaves the test set dominated by samples with high similarity to the training set. The performance of models is severely degraded on samples with lower similarity to the training set but the drawback is highly overlooked in current evaluation. As a result, the performance can hardly be trusted when the model meets low-similarity samples in real practice. To address this problem, we propose a framework of similarity aware evaluation in which a novel split methodology is proposed to adapt to any desired distribution. This is achieved by a formulation of optimization problems which are approximately and efficiently solved by gradient descent. We perform extensive experiments across five representative methods in four datasets for two typical target evaluations and compare them with various counterpart methods. Results demonstrate that the proposed split methodology can significantly better fit desired distributions and guide the development of models.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper studies the problem of generalization in the problem of predicting of drug target binding for drug repurposing, where the affinity score to a specific target is computed on a library of drugs. This paper first draws attention through empirical results to the fact that good performance on a testing set which low-similarity points are underrepresented might be misleading, as the trained model might fail in a real-life setting. Second, the paper introduces an approach to splitting the training and testing subsets for the training of models, that aims at achieving a target distribution on points in the testing set, that might be less biased for the evaluation of a prediction model.

### Strengths
- Clarity: The submission is experimentally rigorous, and well-written. 
- Quality: Several interesting claims are made and illustrated by relevant experiments (for instance, that performance on the testing set is correlated with the proportion of low-similarity samples and that those are often a small part of the dataset). The tested baselines are relevant, and the impact of all relevant hyperparameters is appropriately assessed.
- Significance: The question is important, in particular for healthcare applications, and the algorithmic contribution is valuable, as it is a tractable approach (optimization-wise). 
- Originality: The algorithm proposed to splitting data can handle any target distribution.

### Weaknesses
- Clarity: The section on related works should be present earlier in the paper. 
- Quality: The results shown are not easily reproducible, as the code is not shared. Some aspects to make the method truly appealing to the end users (training for prediction models) are missing: the computational cost of the approach (the computational resources needed to run the experiments), the empirical runtime on the data set of approximately 4,000 samples for instance, and how it compares for these aspects to the selected baselines (random, scaffold, SIMPD). The impact of the number of bins K on the closeness between solution distribution and target distribution is not discussed. 
- Significance: There seems to be a significant gap between the target distribution and the one returned at the end (Figure 6 : 20% of points are present in bins that should remain empty), which might be due to the several approximations made to obtain a tractable optimization problem. I assume that the approach to obtain this figure is to set to zero the number of elements expected in bins associated with similarity ranges higher than the prescribed threshold, but this is not specified in the paper. Finally, the approach relies on the precomputation of a similarity matrix that might be too large to be stored in-place. All the data sets in the paper are of size less than 5,500, but in pharmacompany (private) drug libraries for early drug discovery, there might be 200,000 to $10^6$ compounds [1].

[1] Hughes, James P., et al. "Principles of early drug discovery." British journal of pharmacology 162.6 (2011): 1239-1249.

Even though I rated the paper 8, I think the reproducibility is the largest issue and the absence of code is not justified in the paper. This is a major concern to me.

### Questions
- Could you provide the code for your method or explain why it is not included? Having access to the implementation would greatly aid reproducibility.
- People often split the data into three subsets: training and validation sets (to apply the method on the training set and use the metrics computed on the validation set to update parameters) and testing set (to evaluate the performance of the method). In that case, should the SAE method be applied iteratively? As it is, it can only split data into two groups. 
- For Figure 6, could you specify the exact target distribution used to generate these plots, including any constraints or parameters?

Summary of suggested modifications:
- Please provide experimental code or justify why it is not included. 
- Mention how the approach could be adapted for three-way splits, or what modifications would be needed to support this common scenario.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper targets the performance evaluation of drug-target binding affinity prediction, focusing on analyzing effective data partitioning into training and validation sets. The paper proposes a similarity-aware evaluation (SAE) as an alternative to the conventional random split approach. Recognizing that the similarity of test set molecules to those in the training set significantly impacts machine learning prediction performance, the paper first categorizes validation set molecules into three groups based on their similarity scores with training set molecules. Through extensive experimentation, the paper demonstrates (1) that with random splitting, a large portion of the validation set is occupied by molecules similar to those in the training set, regardless of the feature representation used, and (2) that the overall prediction accuracy aligns closely with the prediction accuracy for training-set-similar molecules dominating the validation set.

The paper argues that, in real-world external test scenarios, there is no inherent focus on molecules similar to those in the training set, and hence, an evaluation approach that is independent of similarity distribution is preferable. However, this cannot be achieved through traditional random splitting. The paper then introduces SAE as a sampling strategy that considers similarity bins and optimizes the scaled distance between the similarity distribution within each bin and a uniform distribution, achieving combinatorial optimization. Given the practical challenges of this discrete combinatorial optimization, the authors propose an approximate solution using continuous relaxation.

The experimental results indicate that SAE’s continuous relaxation-based sampling yields a uniform similarity distribution in the test set, providing a more accurate estimate of model performance in testing. Additionally, the paper demonstrates that when the distribution of an external test set is known, the optimization can be adapted to align with this specific distribution instead of a uniform one.

### Strengths
- The paper is highly readable, with well-organized arguments, making it a thought-provoking study not only for the specific task of DTA but also for general evaluation considerations in molecular machine learning.
- Instead of focusing on prediction methods, this paper addresses the critical issue of how machine learning predictions should be evaluated appropriately. For tasks like drug-target binding affinity (DTA) prediction, practical utility is the key metric, and the gap between benchmarking scores and real-world applicability has been a long-debated issue. It is valuable that such research is presented at venues like ICLR.
- In standard molecular task benchmarking, prediction performance is often evaluated using random split (or scaffold split, as also compared in the paper) for dividing training/validation/test sets. The authors support with data that if an evaluation independent of similarity to the training set is desired, current evaluation practices of using random split or scaffold split may yield an overly optimistic assessment.
- Specifically for the DTA task, the authors empirically demonstrate that regardless of the similarity metric, dataset, prediction target (e.g., IC50 or Ki), or molecular feature representation, as long as random split is used, (1) molecules similar to those in the training set will dominate the validation set, and (2) consequently, only the prediction accuracy for training-set-similar molecules affects the overall performance score. The prediction accuracy for low-similarity molecules, which are fewer in number, is often overlooked. However, considering the actual needs in DTA, this point cannot be ignored.
- As an alternative to the random split, the authors propose a similarity-aware evaluation (SAE), defining it as a combinatorial optimization problem that considers similarity bins and aims to create a uniform distribution within each bin. They also propose a practical solution for SAE using continuous relaxation to approximate the discrete combinatorial optimization.

### Weaknesses
- The technical novelty of the work is relatively limited, as the proposed SAE is essentially a specific case of "stratified sampling" from statistics. The "similarity bins" that underpin SAE effectively serve as strata defined by similarity, meaning that if the dataset is pre-divided into strata based on molecule similarity, applying random sampling within each stratum would naturally achieve uniformity across similarity bins. Therefore, it seems feasible to implement this approach more directly without relying on the combinatorial optimization and continuous relaxation proposed by the authors. From this perspective, the combinatorial optimization formulation may appear to overly complicate the problem.
- In molecular machine learning, including general QSAR tasks, the challenge of fair predictive evaluation has been a longstanding issue. For instance, it was highlighted as a concern for inadequate QSAR model validation over a decade ago [1]. Additionally, this challenge is closely related to out-of-distribution (OOD) generalization [2], and thus it is not exclusive to the DTA task. In [2], a similar analysis to this paper is presented, introducing a "deployment-to-train" distribution-based splitting protocol that resembles the approach proposed in this paper. Note that [2] does not propose any new splitting method, which does not diminish the contribution of this paper.
  - [1] Cherkasov A, Muratov EN, Fourches D, Varnek A, Baskin II, Cronin M, Dearden J, Gramatica P, Martin YC, Todeschini R, Consonni V, Kuz'min VE, Cramer R, Benigni R, Yang C, Rathman J, Terfloth L, Gasteiger J, Richard A, Tropsha A. QSAR modeling: where have you been? Where are you going to? J Med Chem. 2014 Jun 26;57(12):4977-5010.
  - [2] Tossou P, Wognum C, Craig M, Mary H, Noutahi E. Real-World Molecular Out-Of-Distribution: Specification and Investigation. J Chem Inf Model. 2024 Feb 12;64(3):697-711. doi: 10.1021/acs.jcim.3c01774.

### Questions
Have you tried a direct approach like this (stratified sampling), where the dataset is first divided into K bins based on similarity to other molecules (stratification), and then random sampling (random splitting) is applied within each bin to create a test set that is uniform at  similarity-based bins?

### Soundness
3

### Presentation
4

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
This paper addresses a critical limitation of evaluation schema in current drug-target affinity (DTA) prediction models. Traditional methods often employ a randomized dataset split, leading to test sets dominated by samples similar to the training set. This approach causes inflated model performance on high-similarity samples while overlooking the degradation in performance on low-similarity samples, which are common in real-world scenarios. To address this, the authors propose a novel "Similarity Aware Evaluation" (SAE) framework, which introduces an optimized splitting methodology to better match target distributions across similarity bins. This paper demonstrates that SAE enables a more accurate assessment of model generalizability, thereby enhancing performance on external test sets and improving robustness in real-world applications.

### Strengths
- The paper introduces the Similarity Aware Evaluation (SAE) framework, an innovative methodology for creating test sets that better represent real-world data distributions. 
- The authors conduct extensive experiments across multiple datasets and state-of-the-art models, demonstrating the widespread impact of similarity bias.

### Weaknesses
- The continuous relaxation and hyperparameter tuning required in the SAE framework might introduce additional computational complexity and practical challenges for large datasets or high-throughput applications. Could you provide further clarification on the time complexity of the SAE framework, especially in terms of how these processes impact scalability and efficiency in such contexts?
- The performance of the proposed SAE framework likely varies depending on the similarity measurement used, such as Tanimoto or cosine similarity, and may also be influenced by the choice of fingerprint (e.g., ECFP, MACCS, rdikit). A discussion comparing the performance of the SAE framework across these different similarity measures and fingerprints would provide a more comprehensive understanding of its effectiveness.
- While the paper compares SAE with randomized and scaffold splits, it could benefit from a more detailed analysis against other existing methods tailored for similarity-based evaluations, such as time-based, stratified, and cold-drug splits. To provide a clearer benchmark, it would be helpful to include a comparison table showing the performance of SAE against these methods on a common dataset. Additionally, comparisons with specific methods [1,2,3] could be included, or an explanation could be provided for why certain comparisons may not be applicable. This would allow for a more comprehensive evaluation of SAE's effectiveness relative to existing approaches.

[1] Pahikkala, Tapio, et al. "Toward more realistic drug–target interaction predictions." Briefings in bioinformatics 16.2 (2015): 325-337.

[2] Nguyen, Tri Minh, Thin Nguyen, and Truyen Tran. "Mitigating cold-start problems in drug-target affinity prediction with interaction knowledge transferring." Briefings in Bioinformatics 23.4 (2022): bbac269.

[3] Atas Guvenilir, Heval, and Tunca Doğan. "How to approach machine learning-based prediction of drug/compound–target interactions." Journal of Cheminformatics 15.1 (2023): 16.

### Questions
Please see the weaknesses.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper focus on analysing the evaluation methods on Drug-Target binding Affinity (DTA) prediction. The paper first shows that state-of-the-art methods for drug target affinity prediction tend to perform better on drugs that are similar to the drugs used to train the models, than on those with a lower similarity. Authors also shows that current sampling strategies used for the evaluation of DTA tend to report a performance value that is much closer to the similar drugs in training, while the performance for dissimilar drugs is much worst. 

Authors then propose a split methodology called Similarity Aware Evaluation (SAE). The proposed split strategy is capable of reflect any desired distribution, by formulating the problem as an optimisation problem, that can be approximated by gradient decent.

### Strengths
The paper presents thorough evaluations on current SOTA methods for DTA, showing that performance is quite skewed by similar drugs used for training. It clearly illustrates the difference in performance at different levels of drug similarity, under different representations and with several similarity measures.

The proposed similarity aware evaluation and the math behind it is well explained and overall easy to understand. 

The experiments to validate the proposed evaluation are reasonable. The influence and impact of the parameters controlling it are reasonably justified.

### Weaknesses
The paper needs to be proof read for English. While the main ideas in the mathematical part can be understood, the paper writing needs to be improved. (too many missing articles, for example)

While the results presented by the authors do indeed show that current DTA prediction methods tend to perform better on drugs with high similarity comparing to the ones used for training and poorly on the ones with low similarity, this fact is hardly novel or surprising. The influence and challenges of similarity in drug related tasks has been previously documented, but not referenced in the paper. In QSAR related literature this is a known fact, and other areas predicting quantitative properties of drugs do control for this. For example:

Sheridan, Robert P., et al. "Similarity to molecules in the training set is a good discriminator for prediction accuracy in QSAR." Journal of chemical information and computer sciences 44.6 (2004): 1912-1928.

In other areas, such as Drug-Target Interaction prediction this is also often reported by measuring performance on dissimilar drugs in the testing set, for example:

Luo, Yunan, et al. "A network integration approach for drug-target interaction prediction and computational drug repositioning from heterogeneous information." Nature communications 8.1 (2017): 573.

Wan, Fangping, et al. "NeoDTI: neural integration of neighbor information from a heterogeneous network for discovering new drug–target interactions." Bioinformatics 35.1 (2019): 104-111.

This is similar to what authors report on Figure 1 on the left column. All being said, authors do a good job quantifying and reporting the difference in performance with respect to the similarity. 

The related works section itself is reduced and quite compressed, but most of the related work is covered in the introduction.

In the abstract and in the text, the use of the word “target” in the sentence “target distribution” is confusing in the context of this paper, since it could be easily confused with “target protein distribution”.

### Questions
What exactly do authors mean when writing "samples are allowed to coexist in trainset and testset"? (lines 84 and 311) One possible interpretation of this sentence could be that a sample can be used for training the model and later also used to validate the performance. This seems not to be the right interpretation given several parts of the mathematical section of the paper, yet this sentence made me wonder if this is indeed the case. 

In terms of reporting performance, what are the advantages of using SAE over reporting the performance on the different similarity brackets (as done in Figure 1, left column)? This would allow to better pick the model that better suits the similarity bracket instead of having one model for all the drugs.

Related to the previous question, it would be interesting to see the performance on the different brackets in the external dataset of figure 5. Does using SAE improves the performance across all the similarity brackets?

The research topic could also be extended to other QSAR tasks, and while not needed, a small commentary about this could enrich section "4.3 Other Applications".

Do authors consider providing the code implementing SAE, and the code to replicate the experiments?

### Soundness
3

### Presentation
2

### Contribution
3

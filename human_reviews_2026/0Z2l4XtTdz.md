# PAS: Estimating the target accuracy before domain adaptation

- Avg Score: 4.67
- Decision: Accept (Poster)
- Scores: 4, 8, 2

## Abstract
The goal of domain adaptation is to make predictions for unlabeled samples from a target domain with the help of labeled samples from a different but related source domain. The performance of domain adaptation methods is highly influenced by the choice of source domain and pre-trained feature extractor. However, the selection of source data and pre-trained model is not trivial due to the absence of a labeled validation set for the target domain and the large number of available pre-trained models. In this work, we propose PAS, a novel score designed to estimate the transferability of a source domain set and a pre-trained feature extractor to a target classification task before actually performing domain adaptation. PAS leverages the generalization power of pre-trained models and assesses source-target compatibility based on the pre-trained feature embeddings. We integrate PAS into a framework that indicates the most relevant pre-trained model and source domain among multiple candidates, thus improving target accuracy while reducing the computational overhead.
Extensive experiments on image classification benchmarks demonstrate that PAS correlates strongly with actual target accuracy and consistently guides the selection of the best-performing pre-trained model and source domain for adaptation.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper studies an interesting and important problem in transfer learning, specifically unsupervised domain adaptation (UDA) — the assessment of source model transferability. To address this, the authors propose a Potential Adaptability Score (PAS), which measures the compatibility between source and target domains based on the generalization capacity of pre-trained models. PAS leverages pre-trained feature embeddings to quantify the potential adaptability across domains. Although the proposed metric is conceptually interesting and somewhat novel, a major concern lies in the limited discussion and comparison with existing transferability indicators. The authors claim that no comparable baselines exist. However, several existing indices, though possibly less effective, could serve as meaningful references, for example feature distribution discrepancy measures (Maximum Mean Discrepancy, Wasserstein distance) and prediction uncertainty.

### Strengths
-	The paper is well organized and clearly written.
-	The paper targets transferability in domain adaptation, which is both interesting and relevant to the ICLR community.
-	The proposed PAS index provides a potentially useful tool for estimating transferability in UDA scenarios.

### Weaknesses
-	Some claims in the manuscript appear overstated. For instance, Lines 108–109 state that “this is the first proposal for transferability estimation for domain adaptation.” However, there exist several prior works on transferability estimation, such as NCE, LEEP, and LogME. While these studies are mentioned in the related work section, the claim of being the first proposal is inaccurate, as these methods also target transferability and can be adapted for domain adaptation settings.
 
-	Another overstatement appears in the experimental design section (Line 313), where the authors argue that “there is no comparable baseline.” This assumption is not entirely fair. Several existing metrics, such as Maximum Mean Discrepancy, Wasserstein distance, and the averaged prediction entropy over target data, could serve as comparable baselines to evaluate the proposed PAS. Including these in the experiments would make the comparison more convincing and substantiate the claimed superiority of PAS.

-	The paper would benefit from a deeper analysis, particularly in cases where PAS fails to identify the most suitable pre-trained model. Discussing such failure cases could provide valuable insights into the limitations and applicability of the proposed metric.

### Questions
Please refer to the weakness section.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper proposes a new method to estimate the right selection of pre-trained checkpoints and source domains for achieving highest accuracy on a particular target domain. Specifically, they rely on the ratio of the distance between a target samples and the nearest and second-nearest source prototype as a measure of transferability. Though simple, this measure is very intuitive and equally effective in the experiments. Results across various domain transfers show the string correlation between PAS score and eventual DA accuracy.

### Strengths
1. The paper addresses a very important problem of estimating the right choice of source domain and pre-trained model + architecture while faced with optimizing accuracy on an unlabeled target domains with significant domain shift. 

2. The presented hypothesis is very simple and easily usable in wide variety of scenarios indicating the strong effectiveness of the approach. 

3. The distance measure presented in Eq 3 will be small if either (i) the target sample is equidistant from two source classes (poor choice of source domain), or (ii) the two source classes are very close to each other (poor discriminability of source features), covering both the choice for source domain and source pre-trained model efficiently well. 

4. The strong correlation of PAS score with various transfer settings verifies the effectiveness of proposed hypothesis.

### Weaknesses
1. The experiments only consider very basic transfer datasets which are relatively small scale for modern DA. Larger-scale datasets like DomainNet [1] and GeoNet [2] are not covered limiting the insights we can draw from this work for practical use-case settings. 

2. There are no sufficient ablations provided in the experiments regarding other potential measures of transferability. For instance, one measure could just be the average distance between the feature representations of source and target domains in the feature space, and improvements over this simple baseline setting could be helpful. 


[1] Peng, Xingchao, et al. "Moment matching for multi-source domain adaptation." Proceedings of the IEEE/CVF international conference on computer vision. 2019.
[2] Kalluri, Tarun, Wangdong Xu, and Manmohan Chandraker. "Geonet: Benchmarking unsupervised adaptation across geographies." Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition. 2023.

### Questions
1. The experiments currently do not include the choice of which source domain is most suitable for a given target domain, although this is presented as one of the motivations in the introduction. This ablation will be helpful to have in the paper. 

2. It is not completely clear how the centroid of the features are calculated in Eq1. Isn't centroid just the average of the vectors? Why is there an additional normalization in the denominator?

3. As it is mentioned that this method can also be used to choose which source domain matches a particular target domain, it would also be helpful to show why we can't just use _all_ the source domains instead of specifically choosing the best. 

4. It is not clear if this method will be applicable to open-world domain adaptation methods and what modifications need to be performed to adapt this to a more general setting.

### Soundness
4

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
The paper proposes a simple score to predict how well a pre-trained model will perform when adapted to a new domain. The score is based on distances to class centroids. Experiments show that the score correlates with target domain performance.

### Strengths
* The proposed method is clearly explained, and should be easy to implement.
* Experiments show a good correlation between PAS score and target accuracy.
* The experiments use standard datasets for domain adaptation.
* I did not find any spelling or grammar errors.

### Weaknesses
* There is no comparison to related work.
  * The paper claims that "To the best of our knowledge, this is the first proposal for transferability estimation for the domain adaptation setting." But there have actually been investigations into this before. In particular there is the Proxy A-distance (Shai Ben-David, John Blitzer, Koby Crammer, and Fernando Pereira; Analysis of representations for domain adaptation; NIPS 2006), and there should be a comparison with this A-distance.
  * The paper also makes no comparison to methods that first perform domain adaptation and then try to estimate performance, because they claim that these methods are too costly. However, several domain adaptation methods only require a single evaluation of the source model, followed by a relatively cheap adaptation procedure. This should be comparable in cost to the proposed PAS score. An experimental comparison with these methods must be made, which can include a comparison of the computational cost.
* Figure 3 combines multiple datasets, to show a correlation between PAS score and target accuracy. But there are large differences between the different datasets, while the within dataset correlation is perhaps less clear. It is not appropriate to compute a single correlation coefficient for the combined datasets.

### Questions
* What correlation coefficient is used in the paper? Spearman's rank correlation would be more appropriate than Pearson's correlation, because the relation between PAS scores and target accuracy might be nonlinear, and the order is probably most important.
* Why is the PAS score using (d2-d1)/d2? I don't see any motivation for this choice.
* Table 1: What domain adaptation method is this using? Or are you simply applying model on target?
* In my opinion, the pseudo code in Algorithm 1 doesn't add much to the paper, since the explanation in Eq (1)-(3) is much more clear, and the pseudo code does not match how the algorithm would be implemented in a modern framework. In addition, the placement in the paper makes the surrounding text harder to read.

### Soundness
3

### Presentation
2

### Contribution
2

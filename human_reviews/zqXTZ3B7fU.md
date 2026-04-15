# Simplicial SMOTE: Oversampling Solution to the Imbalanced Learning Problem

- Decision: Withdrawn (Treated as Reject)
- Scores: 8, 3, 5, 3

## Abstract
SMOTE is the established geometric approach to random oversampling to balance classes in the imbalanced classes learning problem, followed by many extensions. Its idea is to introduce synthetic data points of the minor class, with each new point being the convex combination of an existing data point and one of its k-nearest neighbors. This could be viewed as a sampling from the edges of a geometric neighborhood graph. Borrowing tools from the topological data analysis, we propose a generalization of the sampling approach, thus sampling from the simplices of the geometric neighborhood simplicial complex. That is, a new point is defined by the barycentric coordinates with respect to a simplex spanned by an arbitrary number of data points being sufficiently close, rather than a pair. We evaluate the generalized technique which we call Simplicial SMOTE on 23 benchmark datasets, and conclude that it outperforms the original SMOTE and its extensions. Moreover, we show how simplicial sampling can be integrated into several popular SMOTE extensions, with our simplicial generalization of Borderline SMOTE further improves the performance on benchmarks datasets.

## Human Reviews

## Human Reviewer 1

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper considers the problem of supervised learning with a class imbalance and proposes a new algorithm for oversampling the minority class based on topological data analysis. The method, Simplicial SMOTE, forms a simplicial complex from data in the minority class, and then generates synthetic samples based on the convex combinations of points sampled from simplices of this simplicial complex. The authors additionally propose variants of other SMOTE algorithms based on simplicial complexes. The utility of the simplicial SMOTE algorithm is validated on synthetic and real datasets, and suggests the value of the approach for a wide variety of empirical applications.

### Strengths
The paper is extremely well presented and provides an original application of topological data analysis in a machine learning setting.

The quality of the work is generally high: The empirical results are presented over a wide set of synthetic and empirical datasets with class imbalances which provide a full picture of the proposed algorithm's value.

The proposed algorithm is presented in an extremely clear way and the figures help to highlight why this approach is different than prior methods. Overall, the presentation is excellent and the paper was enjoyable to read.

The work ultimately provides a valuable step in using topological data analysis (TDA) for machine learning: TDA methods are typically computationally intensive (as noted by the authors), and are quick to be dismissed in machine learning applications. However, this paper shows that TDA methods can still add value to empirical performance of learning algorithms and hence provides a foundation for a wide variety of future work.

### Weaknesses
Some minor points:

1) Although the paper is quite interesting from the lens of topological data analysis, it is presented as a simplicial extension of SMOTE and hence feels limited in terms of significance for a machine learning audience. 

2) The authors could be a bit more clear on why a simplicial complex may be better than a graph for creating synthetic points for oversampling-- it feels like there is some type of local decision-boundary type of argument which would make clear when this method should be valuable.

3) There empirical results could be presented better. The tables are fine and are valuable because they give access to the raw data. However, they could benefit from the addition of confidence intervals. Alternatively, additional visualizations may more clearly summarize the value of the proposed algorithm for each dataset.

### Questions
N/A -- it may be helpful if the authors can address point number 2 above.

### Soundness
4 excellent

### Presentation
4 excellent

### Contribution
3 good

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper introduces a oversampling technique called Simplicial SMOTE to address
the issue of class imbalance in datasets. Diverging from the conventional SMOTE
technique, Simplicial SMOTE innovates by sampling from simplices of a
geometric neighborhood simplicial complex, as opposed to sampling from the
edges of a geometric neighborhood graph. Through evaluation on the benchmark
datasets, it is concluded that Simplicial SMOTE outperforms the original SMOTE and
its variants.

### Strengths
1. Simplicial SMOTE offers an innovative geometric oversampling method to address class imbalance, utilizing topological data analysis tools to extend the capabilities of traditional SMOTE.

2. The method is thoroughly evaluated on a number of benchmark datasets, showcasing its effectiveness and outperformance over existing methods.

3. The paper exhibits clear structure, and high-quality visuals. The writing is clear.

### Weaknesses
1. Increased Computational Complexity: The additional steps of computing simplicial complexes and the requirement for parameter tuning (e.g., maximal simplex dimension) could lead to higher computational complexity, potentially limiting the scalability of the Simplicial SMOTE method, especially for high-dimensional or large datasets. This complexity might hinder the real-time or practical applications of the proposed method in scenarios where computational resources or time are constrained. 

2. Limited Evaluation on High-Dimensional Datasets: The paper evaluates the proposed Simplicial SMOTE method on 23 benchmark datasets, but it does not provide a thorough evaluation on high-dimensional datasets. The behavior and performance of the method in high-dimensional spaces could be different, and it's crucial to understand how the method scales with dimensionality.


3. Parameter Tuning: The necessity for grid search over the maximal simplex dimension p could be seen as a drawback since it adds an extra layer of complexity to the model tuning process. This could potentially lead to longer setup times before the model can be deployed, especially in a production environment.

4. Overall, the novelty of the proposed method is limited.

### Questions
1. How sensitive is the performance of Simplicial SMOTE to the choice of parameters such as the neighborhood size parameter k and and the maximal simplex dimension p ? How were the optimal parameters selected for each dataset in the evaluation?

2. Could the authors provide more insights into the scalability of the proposed method with respect to the number of data points and dimensionality of the datasets? Have the authors tested the method on datasets with larger dimensionality to understand the impact on computational resources and performance?

3. Can the authors provide more details on the computational complexity of Simplicial SMOTE in comparison with the original SMOTE, especially concerning the time and memory requirements? Are there any optimizations suggested to mitigate the computational demand, especially when applying Simplicial SMOTE on large-scale or high-dimensional datasets?

4. Could the authors provide more detailed implementation information to aid reproducibility, such as the specific configurations or hyperparameters used during the evaluation? 

5. Have the authors considered comparing Simplicial SMOTE with more recent methods or extensions of SMOTE, other than the ones mentioned in the paper? How does Simplicial SMOTE compare with state-of-the-art methods in handling imbalanced datasets?
How can Simplicial SMOTE be extended to handle multi-class imbalanced datasets? Have the authors considered evaluating the method in multi-class scenarios?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposed a generalization of the sampling approach to SMOTE, i.e., sampling from the simplices of the geometric neighborhood simplicial complex. The novelty is the barycentric coordinates concerning a simplex spanned by an arbitrary number of data points being sufficiently close rather than a pair. In the experimental section, the authors evaluate the generalized technique, Simplicial SMOTE, on 23 benchmark datasets.

### Strengths
1. The imbalanced classification problem is an interesting and valuable topic in the learning community.

2. The literature part is clear.

3. The structure of the paper is easy to follow.

4. There is extensive experiment analysis on the algorithm performance.

5. The simplicial SMOTE technique can be used to generalize most of the existing types of SMOTE methods.

### Weaknesses
1. In the setup section (section 3, p3), it lacks the assumptions and descriptions on the data distribution (x,y), and especially the level of class imbalance. Without data distribution assumptions, it will limit the guidance for practitioners. 

2. There is no analysis of the theoretical guarantee of the algorithm's performance.

3. The proposed algorithm is more complicated and slower than the baseline algorithms (see Table 1). However, the time performance of the proposed and baseline algorithms is not shown in the paper. Without time performance, it's hard to judge the tradeoff between time and accuracy in the experimental comparisons.

### Questions
1. What's the time performance of the proposed and baseline algorithms?

2. Are there any assumptions and limits on the level of class imbalance?

3. Are there any assumptions on the data distribution to implement the simplicial SMOTE?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper presents a new extension of SMOTE algorithm for binary imbalanced data that instead of generating new instances as linear interpolations, samples from barycentric coordinated from the constructed simplicial complex. The approach is experimentally tested on 23 real datasets and 4 artificial ones. The results show that variants of the proposed method obtains the lowest average ranks on MCC and F1 measures.

### Strengths
- The use of topological analysis (TA) to expand SMOTE is interesting and original. It's probably one of the first works using TA for imbalance problems
- The proposed extension can be also applied to other methods than original SMOTE e.g.  ADASYN, Safe-level SMOTE and Border-line SMOTE

### Weaknesses
- Lack of motivation. It is hard to say what research questions are being asked and answered. The main motivation is that the original SMOTE uses neighbourhood relation arity = 2 and the proposed method >2. It is not clear why increasing the arity will help to classify imbalanced data better. Therefore, it's difficult to assess how the paper advances our understanding of the imbalanced learning problem (other than proposing a slightly better performing method).
- The method is currently not suitable for handling multi-class data which is more frequent in practice. The method is tested only on binary classification problems.
- The paper claims to extend several other SMOTE variants like ADASYN, Safe-level SMOTE and Border-line SMOTE but only the latter is actually tested in the experiments.
- Related works. 
     - The authors divide the imbalanced learning methods into the rather non-standard three types of methods 1) cost-sensitive 2) under-sampling 3) over-sampling. This categorisation does not include the methods that combine oversampling with undersampling or specialized ensemble methods. - see the book of Fernández at al. "Learning from Imbalanced Data Sets"
  - In the description of the SMOTE method, which is at the focus of this paper, the authors only say that "the new synthetic points as the random convex combinations of pairs consisting of a point and its nearest neighbor", leaving out the rather important information that only neighbours from the same class are considered. This also makes the comparison of Neighbourhood Size/Relation arity in Table 1 a bit misleading, since e.g. Mixup and SMOTE are compared, but the former uses neighbours from the whole dataset and the latter only from the selected class. 
  -  In general, the original Mixup (used in the paper) is not a technique addressing class imbalance. Therefore, the Table 1 providing motivation for the approach by comparing it to others, actually compares the proposed approach with SMOTE, non-imbalance learning technique Mixup and Random Oversampling (which do not really use any neighbourhood graph) 
  - The authors divide the data level approaches into local (like ROS) and global (like SMOTE). It's quite difficult to understand what local and global mean in this context. Typically, in the imbalanced learning literature, ROS would be a global method (as it focuses on global class imbalance) and SMOTE would be a more local approach (taking into account the local characteristics of a sample). The term "geometric sampling methods" is also newly introduced by the authors and is used, among other things, to refer to random oversampling that do not take geometric relationships into account.
- Clarity. The concept of "sampling from the edges of the neighborhood graph" introduced in the introduction is not very clear to me, and it is not cited. In general, I find the two last paragraphs of the introduction quite difficult to read and the paper would benefit from some more intuitive description. 
- Experiments 
     - It's hard to say what was the purpose of the experiment on artificial data, since it is not used to observe specific properties of the approach or demonstrate that the method addresses some issue in an isolated environment. The only conclusion relies on an assumption that "circle inside a circle" is more geometrically complex than "two moons" or "swiss rolls"  which I find questionable.
  - The statistical comparisons are performed only with respect to the original SMOTE from 2002 and not to any of the more modern extensions
  - Lack of evaluation with some state-of-the-art implementation of GBT algorithm like CatBoost or XGBoost
  - F1 score metric has been criticized in imbalanced learning literature. It'd be better to also report G-mean value and other specialized metrics.
  - Making the implementation of the method available would increase the reproducibility of this research and its potential impact.


Typo: "first approximating is using a set of prototype points obtained by LVQ"

### Questions
- How the datasets were selected? The imbalanced-learn library contains 27 datasets but only 23 were selected for the experiments.
- What was the running time of the proposed method and how it compares with other methods under study?
- The authors use Mixup in their experiment which interpolates not only the feature values x but also target vectors y. Such interpolated target vectors are easy to use in NN but not necessarily in GBT or kNN used in the experiments. How it was applied? If no target vector interpolation was used, how the target value was established?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

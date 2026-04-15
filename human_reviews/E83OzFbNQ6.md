# Online Continual Learning Without the Storage Constraint

- Decision: Withdrawn (Treated as Reject)
- Scores: 5, 5, 3

## Abstract
Traditional online continual learning (OCL) research has primarily focused on mitigating catastrophic forgetting with fixed and limited storage allocation throughout an agent's lifetime. However, a broad range of real-world applications are primarily constrained by computational costs rather than storage limitations. In this paper, we target such applications, investigating the online continual learning problem under relaxed storage constraints and limited computational budgets. We contribute a simple algorithm, which updates a kNN classifier continually along with a fixed, pretrained feature extractor.  We selected this algorithm due to its exceptional suitability for online continual learning. It can adapt to rapidly changing streams, has zero stability gap, operates within tiny computational budgets, has low storage requirements by only storing features, and has a consistency property: It never forgets previously seen data. These attributes yield significant improvements, allowing our proposed algorithm to outperform existing methods by over 20\% in accuracy on two large-scale OCL datasets: Continual LOCalization (CLOC) with 39M images and 712 classes and Continual Google Landmarks V2 (CGLM) with 580K images and 10,788 classes, even when existing methods retain all previously seen images. Furthermore, we achieve this superior performance with considerably reduced computational and storage expenses.

## Human Reviews

## Human Reviewer 1

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
In this study, the authors delve into online continual learning under the constraints of limited computational resources, while relaxing storage capacity constrain. The central concept revolves around the integration of a K-nearest neighbors (KNN) approach with a static, pre-trained feature extractor. The authors substantiate the appropriateness of KNN for this scenario, underscoring its capability to swiftly adapt to dynamic data streams without compromising stability. Furthermore, they underscore its efficiency in the context of constrained computational resources, attributed to its ability to store essential features exclusively and uphold a consistency property, thereby preserving previously encountered data.

### Strengths
In this paper, online  continual learning in the presence of drift is studies, which indeed is an interesting and a practical topic as data streaming applications keep on increasing. The paper is well written and easy to read, and the algorithm is clearly presented and also demonstrated in Figure 1. The paper very well included state of the art. The improvement obtained in the experiments is considerable.

### Weaknesses
In my opinion, the assumed setup seems simplified and unrealistic, given that it presumes the use of a fixed pre-trained feature extraction method for all forthcoming data in the data stream, as also mentioned by the authors. In a data stream, the data distribution and their features structure can evolve, and new classes can emerge, using a fixed pre-trained feature extraction method might not be enough in data streaming learning. The authors discussed about mobile devices, but can one really claim that we don't have  storage constraints in mobile devices in data explosion era? 
This core idea in this study is very similar to (Nakata et al., 2022), in both studies KNN has been used as a reminder assistance for the backbone model for quick adaptation after drift. The idea is indeed interesting, but unfortunately it does not contain novelty. I cannot find any major novel differentiation within this work in compare with the mentioned study.

### Questions
Can you provide some explanation and comparison with KNN for the approaches mentioned in:"Fixed Feature Extractor based Approaches"
Can you add some justification or empirical results to prove this: "We additionally highlight that ACM has a substantially lesser computational cost compared to traditional OCL methods."

You have used a fixed set of hyperparameters determined from the pretrained set, but did (can) you explore how the need for hyperparameters tuning can vary when observing drift? BTW,  Please mention in the text the concrete hyperparameters here as well: "We first tune hyperparameters of all OCL algorithms on a pretraining set"

Could you please provide experiments or explain how your algorithm in compare with other approaches work when a new class emerges that has not been see in the pretraining phase? 

In KNN have you assumed a fixed K?

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper proposes a kNN based retrieval of previous sample from the infinite memory for reminding the past information to alleviate forgetting. While the setup could be realistic as the memory cost becomes negligible, even using the efficient version of kNN, the retrieval of relevant sample from the infinite memory is still computationally expensive and reminding previously learned knowledge may not be desirable as large models are arguably much less forgettable about previously learned knowledge. By the help of perfect remind of previously given data, the method improves the classification accuracy significantly over the other methods.

### Strengths
- Superior empirical gain over other methods
- Simplicity of the method
- Good empirical setup using CGLM and CLOC datasets

### Weaknesses
- The presented setup with infinite memory is arguably realistic online continual learning setup. The infinite memory would eventually prevent forgetting by perfect reminding (by using properly efficient version of kNN retrieval) and the proposed method is not surprising with that. Thus, it is questionable whether the proposed setup and the method is indeed helping us to solve online continual learning for real world deployment or not.
- Method is not well motivated. It is not clear why the kNN let the model adapt to new sample fast and lead to zero stability gap inherently.
- Why the proposed method only has very high initial accuracy in Rapid adaptation plot using CLOC in Fig. 2 (right upper).

### Questions
See weaknesses.

### Soundness
3 good

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The proposed submission questions whether the popularly enforced storage constraints in online continual learning is really realistic and develop an online continual learning method with no storage constraints, e.g. storing all data points (whether raw or processed). They use an approximate kNN classifier to make predictions using the stored data points, which boosts fewer computation costs in terms of training and predicting using the continually learned model. Evaluation is done on YFCC-100M and Google Landmarks V2 datasets.

### Strengths
- The submission poses an interesting question of whether the storage constraint is realistic or not.
- The proposed method is simple and straightforward.
- The paper reads well.

### Weaknesses
- The novelty is limited. The methodology itself is an approximate kNN, with little modifications. Additionally, the method merely uses pretrained feature extractors as well, which does not add to technical novelty.
- The consideration of the storage constraint seems a bit uni-dimensional to me. There are other factors than just storage costs that are not taken into account. For instance, the data itself may be volatile, i.e., some data points may be required by law to be deleted upon a set duration.

### Questions
- The feature computation part seems to have little difference with regards whether it is used in a regular, static data manner or in the continual learning setup. Can the authors clarify on this?

### Soundness
2 fair

### Presentation
3 good

### Contribution
2 fair

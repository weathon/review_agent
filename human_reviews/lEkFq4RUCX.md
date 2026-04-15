# Directional Distance Field for Modeling the Difference between 3D Point Clouds

- Decision: Reject
- Scores: 6, 8, 5

## Abstract
Quantifying the dissimilarity between two unstructured 3D point clouds is challenging yet essential, with existing metrics often relying on measuring the distance between corresponding points which can be either inefficient or ineffective. In this paper, we propose a novel distance metric called directional distance field (DDF), which computes the difference between the underlying 3D surfaces calibrated and induced by a set of reference points. By associating each reference point with two given point clouds through computing its directional distances to them, the difference in directional distances of an identical reference point characterizes the geometric difference between a typical local region of the two point clouds. Finally, DDF is obtained by averaging the directional distance differences of all reference points. We evaluate DDF on various optimization and unsupervised learning-based tasks, including shape reconstruction, rigid registration, scene flow estimation, and feature representation. Extensive experiments show that DDF achieves significantly higher accuracy under all tasks in a memory and computationally efficient manner, compared with existing metrics. As a generic metric, DDF can unleash the potential of optimization and learning-based frameworks for 3D point cloud processing and analysis. We include the source code in the supplementary material.

## Human Reviews

## Human Reviewer 1

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
5: You are absolutely certain about your assessment. You are very familiar with the related work and checked the math/other details carefully.

### Summary
This paper introduces a new metric for measuring the distances between two point clouds. The idea is to compute the difference between the underlying 3D surfaces calibrated and induced by a set of reference points. The evaluations demonstrates the effectiveness of the proposed DDF loss.

### Strengths
1. The motivation makes sense. Previous losses (e.g. CD, EMD) focus on the point-to-point distances as the supervision, which brings lots of computational cost or easily reaches a local minimum. The proposed DDF loss measures the distance and directions to the underlying surface.

2. The performance seems good, which outperforms the widely used CD, EMD and DCD.

### Weaknesses
1. The name directional distance field is not suitable. I do not understand what is it until I finished reading the method section. The 'field' often indicates the signed distances or occupancies learned by a neural network. The proposed loss to measure distances from a reference point to the underline surface is not a 'field'.

2. The presentation can be improved.  I suggest that the authors to improve the writings in the introduction to make the readers understand the loss more easily. The inappropriate name 'directional distance field' and the unexplained 'reference point' make the introduction not clear enough. I can not understand the loss until I finish reading the method section, but finally I find the loss quite simple.

3. In Fig. 3, I find that DCD achieves quite good performances, why is the quantitative results of DCD in Tab. 1 that bad? More comparisons are needed.

4.  As shown in Fig.4, DDF is less efficient than CD and DCD. What is reason? Since CD measures point-to-point distances which should be much slower.

### Questions
It will be interest to see the performance of DDC under more downstream tasks like point cloud completion, point cloud generation, etc.

### Soundness
3 good

### Presentation
2 fair

### Contribution
3 good

---

## Human Reviewer 2

### Rating
8: accept, good paper

### Rating Number
8

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes a new metric to measure the distance of two 3D point clouds.
The proposed method utilizes reference points to represent the local feature of the surface where point clouds should be. The distance is computed based on those reference points.
Experiments are performed on four downstream tasks, showing the proposed metric improves the performance of the methods of those downstream tasks.

### Strengths
1. It is reasonable to utilize the surface where the point clouds should be to compute the distance of two 3D point clouds.
2.The experiments show the proposed method improves the performance of all the downstream tasks.
3. The implementation of methods of all the downstream tasks are explained in detail.

### Weaknesses
The generation of reference points are not explained very clearly. The reviewer is confused by the shared identical weight operation and the reference point generation process.

### Questions
The reference points are generated from one of the two point clouds, and the weights in Equ.1 are computed for each kNN points and reference points. The reviewer wants to ask how to share the weights in g(q_m, P1) and g(q_m, P2)?

### Soundness
4 excellent

### Presentation
3 good

### Contribution
3 good

---

## Human Reviewer 3

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This work presents a point cloud distance function as an alternative to EMD and CD. The proposed distance is theoretically superior to EMD and CD as it better describes underlying surfaces. The experimental result over several tasks (3D shape reconstruction, rigid registration, scene flow estimation and feature representation) demonstrates this.

### Strengths
I believe the idea of using better surface descriptions for point cloud related tasks is relevant and this paper show improvements in several related tasks including shape reconstruction, rigid registration, scene flow estimation, and feature representation. 

Despite some comments, the distance function is technically sound. 

Paper is generally well presented.

### Weaknesses
This work should add comparisons against relevant shape or surface descriptors in the literature. For example, against 3D shape context (and other methods) in Frome et al. ("Recognizing objects in range data using regional point descriptors." ECCV 2004). This lack of comparison is an important weakness as the proposed method resembles the common pipeline of 
1)  Computing keypoints, here by using a sampling mechanism plus noise;
2) Obtaining descriptors at each keypoint, here as the concatenation of the magnitude and direction of a sum of weighted distances between a keypoint and the K-NN of a point cloud. 

The proposed distance aggregates descriptor distances. Here, descriptor correspondences result from sharing the same keypoints to obtain each set of descriptors. 

Another weakness is in formulations that should be presented more clearly, perhaps improving notation. For example, it is unclear why "g(qm, P1) and g(qm, P2) share identical weights". From equations, they seem to be different as calculated over different point clouds. 

Additional 

In Sec. 4.2,  R should be in SO(3). 

Also, there is some abuse of notation when applying a rotation over a point cloud. 

The registration methods Opt-EMD, Opt-CD, Opt-ARL and Opt-Ours require better explanation.

### Questions
Why do you propose g = [f, v] instead of a 3-vector f*v?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

## Human Reviewer 1

### Summary
The paper replaces Multi-Layer Perceptrons (MLPs) in the PointNet model with Kolmogorov-Arnold Network layers (KANs) to create a point-cloud-based neural network for classification or segmentation tasks on unordered 3D point sets. The proposed approach uses Jacobi polynomials to construct PointNet-KAN and investigate its performance across different polynomial degrees. The paper also includes efforts to examine the effect of special cases of Jacobi polynomials, including Legendre polynomials, Chebyshev polynomials of the first and second kinds, and Gegenbauer polynomials. The paper supports the proposed method with extensive evaluations of PointNet-KAN hyperparameters, such as the degree and type of polynomial used in constructing KANs. PointNet-KAN is a shallower and simpler network compared to PointNet and achieves competitive performance.

### Strengths
+ The proposed approach accommodates Jocbi polynomials and special case polynomials and provides experimental results.
+ The paper is well-written and easy to understand.

### Weaknesses
- The proposed method evaluation is limited to synthetic datasets. Most recent state-of-the-art 3D Classification methods use real-world datasets such as ScanObjectNN. 
- The time complexity of the pointNet-KAN model during training/testing is not discussed. It is unclear if it is worth replacing MLPs with KANs as the overall accuracy improvements are less significant (< 2%).
- The proposed approach is a straightforward replacement of MLPs by KANs. Although it has limited novelty for a paper in the ICLR main track, it would be a good contribution if submitted to one of the workshops.
- While qualitative results for part segmentation are provided for Jacobi polynomials, the results for special polynomials such as Legendre, Chebyshev, and Gegenbauer polynomials are not provided. 
- Table 3 shows that the increase in Jacobi polynomial degree and the corresponding increase in the number of parameters seem to have marginal improvements in both mean and overall accuracy. So, it is unclear if the increase in accuracy due to the increase in Jacobi polynomial degree from 2 to 4 is due to the increase in the number of parameters only and not due to the proposed approach.
- The mean or overall accuracy for classification and segmentation tasks is lower than that of the original PointNet model (refer to Tables 1 and 2).

### Questions
Kindly provide experimental results to back the claim below.

"Using more complex versions of PointNet could introduce other factors that might obscure the direct influence of KANs, making it challenging to determine whether any performance changes are due to the KAN architecture or other network components."

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
3

### Confidence
3

---

## Human Reviewer 2

### Summary
he paper integrates KAN into PointNet to propose a PointNet-KAN. It replaces MLPs with KAN and conducts validations on 3D object classification and part segmentation tasks. Compared with PointNet, the PointNet-KAN performs better on classification but maintains similar segmentation results on the ShapeNet part dataset. Overall, this work validates the KAN on 3D point domains but lacks novelty and contributions.

### Strengths
Integrating KAN into the 3D domain is interesting and may benefit the field of study.

### Weaknesses
1. Limited novelty and contribution.  This paper looks more like a technique report rather than a research paper. It naively uses KAN to replace the MLPs in PointNet without any technique improvements on KAN. It would be more preferable if the paper could adapt KAN in terms of some properties of 3D points, like irregularity and unorderness, to fit the nature of 3D points.

2. Lack of motivation to introduce KAN into 3D point cloud domain. In the Introduction part, the paper mentions that KAN can learn activation functions by itself but does not illustrate other benefits of introducing KAN into the 3D domain. Therefore, it is unclear why to integrate KAN into PointNet.

3. Experiment results are not satisfactory and need more improvements. In terms of Tables 1&2, PointNet-KAN can only marginally surpass PointNet on ModelNet classification and is even lower than PointNet on part segmentation tasks.These weak results fail to validate the effectiveness of the proposed method.

### Questions
If the paper aims to validate the effectiveness of replacing KAN with MLPs, how about trying more network architectures?

### Soundness
1

### Presentation
2

### Contribution
1

### Rating
3

### Confidence
4

---

## Human Reviewer 3

### Summary
This paper is the first to introduce KAN in the 3D domain, replacing MLP with KAN based on PointNet. Experiments are conducted on both classification and segmentation tasks, providing guidance for the subsequent application of KAN in point cloud analysis tasks.

### Strengths
This paper is the first to apply KAN to point cloud analysis tasks. The article is logically coherent, with complete formulas, and includes a substantial amount of experimentation to demonstrate the effectiveness of the proposed method, providing valuable guidance for the future development of KAN in point clouds.

### Weaknesses
1. Some experimental results are concerning and lack detailed discussion or explanation. For instance, on ModelNet40, there is a significant performance difference for PointNet-KAN depending on whether the input includes normal vectors, suggesting that PointNet-KAN is highly sensitive to input variations. However, the authors did not provide a more thorough investigation or explanation regarding this sensitivity. Additionally, on ShapeNet Part, the performance of PointNet-KAN still falls short compared to PointNet. Furthermore, based on subsequent experimental results, it seems that merely increasing the scale of PointNet-KAN does not address this issue.

2. The robustness experiments only consider variations in point quantity and lack experimental results when increasing noise and rotation.

3. For the classification tasks, experiments were only conducted on the ModelNet40 dataset and the PointNet model. It is recommended to include experiments on the ScanObjectNN dataset as well. Additionally, since PointMLP is also based on MLP, it may be worthwhile to explore embedding KAN into PointMLP.

### Questions
1. It is suggested that the authors provide a more detailed discussion and analysis of the experimental results. For example, they should explore why the presence of normal vectors has such a significant impact on the performance of PointNet-KAN on ModelNet40 and why the performance on ShapeNet Part is inferior to that of PointNet.

2. It is recommend that the authors include experiments on the ScanObjectNN dataset to further demonstrate the applicability of PointNet-KAN. Additionally, conducting experiments that address the effects of rotation and noise would provide a more comprehensive assessment of the robustness of PointNet-KAN.

3. Is there more detailed data regarding the $\textbf{influence of the size of tensors and global features}$? This could provide insights into the scalability of PointNet-KAN to some extent.

4. For Figure 2, I suggest adding a visual comparison with other methods to demonstrate the effectiveness of PointNet-KAN.

### Soundness
2

### Presentation
3

### Contribution
2

### Rating
5

### Confidence
4
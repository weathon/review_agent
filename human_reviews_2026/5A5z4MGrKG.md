# Implicit Neural Compression of Point Clouds via Learnable Activation Function

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 4, 4

## Abstract
Efficiently compressing and transmitting large-scale high-fidelity 3D point clouds is a critical bottleneck for practical applications.  We introduce a novel framework that reformulates point cloud compression as model compression. Our framework models high-fidelity point cloud geometry and attribute with compact implicit neural representations (INR) separately and then compresses the model parameters directly via quantization and entropy coding, decoupling representation from compression. To ensure this neural representation is both faithful and efficient, we employ Kolmogorov-Arnold Network (KAN) as the INR backbone. Thanks to its superior approximation properties and parameter efficiency, KAN can easily capture fine-grained details missed by traditional MLP. Extensive evaluations on datasets such as KITTI, ScanNet, and 8iVFB demonstrate that our method significantly outperforms the MPEG standard and prior implicit neural representation approaches. Notably, it achieves competitive rate-distortion performance against state-of-the-art deep learning codecs. Our findings establish implicit neural compression as a powerful and practical pathway for developing the next generation of high-efficiency point cloud codecs.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper introduces Point cloud Implicit neural COmpression (PICO), a framework that transforms point cloud compression from a signal processing issue to a neural network compression problem by modeling geometry and attributes separately with compact implicit neural representations and compressing their parameters via quantization and entropy coding. This approach decouples geometry and attribute modeling to prevent feature entanglement and separates representation from compression for precise control over rate and quality. PICO employs a multi-scale rate control mechanism, using a pre-computed Pareto frontier for coarse-grained architecture selection and tunable L1 regularization for finer-grained parameter sparsity, enabling precise bitrate allocation through quantization step size adjustment. The framework adopts the Kolmogorov-Arnold Network-inspired Learnable Activation Function Network as its INR backbone, which captures high-frequency details with fewer parameters and is further enhanced for PCC with positional encoding and radial basis functions. Evaluations on the 8iVFB, KITTI, and ScanNet datasets demonstrate its superiority over MPEG standards and other PCC methods. The contributions include PICO's precise rate control and real-world optimization, the lightweight and effective LeAFNet backbone, and the reformulation of PCC as neural network compression.

### Strengths
I am fond of the proposed idea.  While I cannot confirm whether an identical approach has been previously applied to point cloud compression, it is worth noting that similar strategies have proven highly effective in the fields of 3D representation and 3D reconstruction/novel view synthesis.  The paper is well-articulated, with most concepts clearly explained.  Furthermore, the simplicity and directness of the algorithm's implementation stand out as a significant advantage.  In the experimental section, the paper provides a relatively thorough analysis of Adaptive Model Parameter selection.

### Weaknesses
My concerns regarding this work are as follows:

1) As indicated by the quantitative comparison experiments, the proposed method does not surpass, and in some cases, slightly underperforms compared to the previous state-of-the-art method, Unicorn. This diminishes my confidence in the method's overall effectiveness.
2) What is the speed performance of the proposed method? The authors should report the compression speed and provide a comparative analysis with other methods. Slow processing speed or low efficiency is widely recognized as a drawback of INR-based approaches, and I am curious to know whether the proposed method suffers from this limitation.
3) The network architecture employed in the paper lacks innovation, as it primarily consists of several generic modules, including Positional Encoding, Quantization & Entropy Coding, and Regularized Training.

### Questions
Please refer to the questions raised in the Weaknesses section. I hope the authors can provide responses to these issues in the rebuttal.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes a point cloud compression method based on implicit neural representations. It first converts the point cloud into a voxel representation, and then trains a neural network to predict the occupancy and attributes of each point. Finally, the point cloud is compressed by coding the parameters of the trained network.

### Strengths
The paper is clearly structured and easy to follow. The proposed method sounds reasonable, and the quantitative results are also good.

### Weaknesses
1. Is the INR trained individually for each point cloud? Or can a single trained network generalize to arbitrary point clouds? If the model needs to be optimized separately for every point cloud, what is the required training time? In that case, its efficiency would be far inferior to feed-forward based methods.

2. In Line 264, it is mentioned that the optimal 𝑡 is searched to maximize the D1 PSNR. Is this search process performed only for the proposed method, or for all baselines as well? If only applied to the proposed method, the comparison would not be fair.

3. The Sampling Strategy section introduces several tricks to reduce training time and memory consumption. I would like to know how much practical improvement they bring — e.g., how many minutes of training are saved, and how much memory is reduced?

4. Why does the paper only provide quantitative results but no visualization results?

5. Prior works [1, 2] typically evaluate the decompressed point clouds on downstream tasks (e.g., object detection on KITTI) to demonstrate effectiveness. This paper lacks such experiments.

6. The paper does not include comparisons with some key baselines [3, 4], even though these baselines provide open-source implementations.

7. Regarding the backbone, how significant is the performance difference between using a KAN-based architecture and using an MLP?

[1] Que, Zizheng, Guo Lu, and Dong Xu. "Voxelcontext-net: An octree based framework for point cloud compression." CVPR, 2021.

[2] Huang, Lila, et al. "Octsqueeze: Octree-structured entropy model for lidar compression." CVPR, 2020.

[3] Fu, Chunyang, et al. "Octattention: Octree-based large-scale contexts model for point cloud compression." AAAI, 2022.

[4] You, Kang, et al. "Reno: Real-time neural compression for 3d lidar point clouds." CVPR, 2025.

### Questions
Please refer to the Weaknesses section for details.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
In this paper, the authors present a point cloud compression framework that employs a Kolmogorov–Arnold Network (KAN) as an alternative to traditional backbone architectures. To enhance computational efficiency, the method partitions the 3D space into subspaces and processes only the occupied voxel blocks. The framework enables variable-rate compression by using models from a predefined model dictionary with varying parameter counts. Based on rate–distortion (RD) curve analysis, the most suitable model is selected for a given compression rate. Geometry and attribute data are compressed separately, with attribute compression dependent on the reconstructed geometry. Additionally, the framework adopts dynamic thresholding to determine voxel occupancy. The authors evaluate their method across three benchmark datasets, demonstrating its effectiveness for efficient, adaptive point cloud compression.

### Strengths
1. The use of the Kolmogorov–Arnold Network (KAN) as the implicit neural representation (INR) backbone demonstrates a solid theoretical motivation, leveraging KAN’s superior approximation capabilities and parameter efficiency.

2. The method is evaluated on multiple benchmark datasets (KITTI, ScanNet, and 8iVFB).

3. The paper maintains good readability and technical clarity throughout.

### Weaknesses
1. The proposed model dictionary appears dataset-specific and may not generalize well to unseen point cloud distributions. Please discuss the potential limitations and any strategies to improve generalization.

2. “we divide the original space S into 2M × 2M × 2M coarse-grained cubes” — It is unclear how boundary issues are handled during reconstruction. Given the independent nature of neighboring blocks, the surface reconstruction may not be seamless. Did the authors observe any boundary artifacts or discontinuities? An additional experimental result illustrating this issue would strengthen the discussion.

3. “For coarse-grained control, we select an optimal model architecture using a pre-computed Pareto frontier that profiles the trade-off between model size and bitrate” — The analysis related to the Pareto frontier is a key design element, but is not clearly presented. Please include the corresponding experiments or plots in the results section to substantiate this claim.

4. A comparison with conventional backbones would better highlight the advantages and significance of LeAFNet.

5. The notation should be made consistent throughout the manuscript (e.g., V - X vs. V\X). Inconsistent notation can confuse readers.

6. Qualitative results should be included to visually demonstrate the reconstruction quality in comparison with existing methods.

### Questions
Mentioned in the weaknesses section.

### Soundness
2

### Presentation
3

### Contribution
2

# NAB: Neural Adaptive Binning for Sparse-View CT reconstruction

- Avg Score: 4.50
- Decision: Accept (Poster)
- Scores: 4, 4, 4, 6

## Abstract
Computed Tomography (CT) plays a vital role in inspecting the internal structures of industrial objects. Furthermore, achieving high-quality CT reconstruction from sparse views is essential for reducing production costs. While classic implicit neural networks have shown promising results for sparse reconstruction, they are unable to leverage shape priors of objects. Motivated by the observation that numerous industrial objects exhibit rectangular structures, we propose a novel \textbf{N}eural \textbf{A}daptive \textbf{B}inning (\textbf{NAB}) method that effectively integrates rectangular priors into the reconstruction process. Specifically, our approach first maps coordinate space into a binned  vector space. This mapping relies on an innovative binning mechanism based on differences between shifted hyperbolic tangent functions, with our extension enabling rotations around the input-plane normal vector. The resulting representations are then processed by a neural network to predict CT attenuation coefficients. This design enables end-to-end optimization of the encoding parameters---including position, size, steepness, and rotation---via gradient flow from the projection data, thus enhancing reconstruction accuracy. By adjusting the smoothness of the binning function, NAB can generalize to objects with more complex geometries. This research provides a new perspective on integrating shape priors into neural network-based reconstruction. Extensive experiments demonstrate that NAB achieves superior performance on two industrial datasets. It also maintains robust on medical datasets when the binning function is extended to more general expression. The code is available at \url{https://github.com/Wangduo-Xie/NAB_CT_reconstruction}.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper offers a clear shape-aware alternative to RFC-based INRs by constructing trainable, rotated bins that better match industrial rectangular priors. Instead of random Fourier features in INRs, NAB encodes each spatial coordinate via differentiable rectangular bins formed by differences of shifted tanh functions along two axes, with learnable translation, scale, rotation, height, and steepness. A small FCN maps these bin features to attenuation values, and training minimizes projection-domain MSE with a differentiable forward operator. The authors analyze a limiting property showing convergence of the soft bins toward hard binary partitions as steepness increases. Experiments report sizeable PSNR/SSIM gains over DIP, Wire, and several INR variants, plus ablations on bin parameters.

### Strengths
- **Shape-aware coordinate encoding.**  
  The paper proposes a shape-aware coordinate encoding scheme tailored to the geometric characteristics of industrial components, effectively leveraging rectangular priors common in real-world applications. For non-canonical rectangles, the model introduces learnable rotation parameters to handle arbitrary orientations, providing greater flexibility. The integration of **multi-scale steepness** further extends the representation from sharp-edged rectangles to smooth, bulge-like bins, enhancing expressiveness.

- **Novel prior design for INR tasks.**  
  Incorporating a shape-aware prior into the coordinate encoding stage of an implicit neural representation (INR) is both meaningful and underexplored. This design introduces a new perspective for geometry-adaptive representation learning and offers potential inspiration for future INR-based industrial CT research.

### Weaknesses
- **Limited dataset diversity and realism.**  
  The evaluation is conducted on only two simulated datasets—**CaCO₃** and **Workpieces**—both of which are relatively simple. The CaCO₃ dataset lacks geometric complexity, and the Workpieces dataset includes only a small number of slices. It would strengthen the paper to include experiments on more complex real-world components and on real projection data.

- **Narrow baseline selection.**  
  The comparative baselines are limited to several RFC-based MLP variants. Given that this paper focuses on a specific industrial CT application, it would be more convincing to include comparisons with **advanced INR methods** such as **SIREN** [1] and **InstantNGP** [2], which are now standard benchmarks for implicit representation learning.
---
Overall, the paper presents an interesting idea with potential impact. However, the experimental validation is not sufficiently thorough. The choice of baselines lacks persuasiveness, and the datasets used appear overly simplistic. Providing more rigorous and comprehensive experiments would considerably strengthen the paper and make the results more convincing. I would be open to reconsidering my assessment if the authors can present more substantial empirical evidence in a future revision.

### Questions
- **On initialization of the neural network.**  
  The appendix states: “For the neural network $f_{\text{net}}$’s parameters, we use the initialization strategy from (Shen et al., 2022).” This is unclear. In NeRP, initialization encodes temporal similarity of the same object. How is this strategy adapted in the current paper? Please clarify how the initialization is performed here.

- **On relation to 3D Gaussian representations.**  
  The proposed shape-aware rectangular encoding appears conceptually similar to explicit 3D Gaussian representations, except that rectangles replace Gaussian ellipsoids. Could the authors add **3D Gaussian representation** as a baseline to directly compare their relative performance?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This work proposes a new reconstruction method based on implicit neural representations (INR), namely NAB, for industrial CT imaging. The main motivation is that most man-made industrial objects are composed of rectangular structures, while existing INR methods fail to exploit such explicit shape priors, leading to suboptimal performance. To address this issue, the proposed NAB introduces a new encoding strategy, where a hyperbolic tangent function is used to form an adaptive binning operation. With this encoding, the shape prior can be implicitly incorporated into the network optimization, resulting in improved reconstruction quality. The experiments also demonstrate the superiority of the proposed NAB over existing INR methods.

### Strengths
- The key motivation and idea of this paper are interesting. Incorporating shape priors is a reasonable way to improve the reconstruction accuracy for industrial CT imaging.
- The proposed encoding strategy (i.e., the adaptive binning operation) is technically sound.
- The paper is clearly written and easy to follow.

### Weaknesses
My main concern lies in the experimental evaluation:
- The compared baselines mainly include the random Fourier encoding, but exclude Instant-NGP [1]. To my knowledge, Instant-NGP is much more powerful than random Fourier encoding in various inverse problems, such as CT or MRI.
- In Table 1, the $INR_{l_1}$ performs worse than $INR_f$, which is a bit strange since the former has more learnable parameters.
- The random Fourier encoding has two key hyperparameters, the mean and the standard deviation, which strongly affect its learning capacity. However, these hyperparameters are not reported in the paper.
- From Figures 5 and 6, the objects from the Workpieces dataset have more complex shapes than those from the CaCO3 dataset. However, the improvement achieved by NAB on the Workpieces dataset is relatively small. Does this indicate that NAB does not perform well on more complex objects?
- In Table 2, the results of $INR_{l_2}$ are missing.
- In Figures 5 and 6, the qualitative results of the second-best method ($INR_{l_2}$) are also missing.

[1] Müller, Thomas, et al. "Instant neural graphics primitives with a multiresolution hash encoding." ACM transactions on graphics (TOG) 41.4 (2022): 1-15.

### Questions
Overall, this paper is interesting in its idea and motivation, but the current experimental evaluation is somewhat limited. Therefore, I cannot recommend acceptance in the current version. I would be willing to raise my rating if the above concerns can be properly addressed.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
The paper proposes a neural adaptive binning scheme as an alternative to traditional random Fourier feature coordinate encoding. The scheme not only supports rigid body transformations but also achieves scaling transformations, which enables explicit modeling of the rectangular priors commonly encountered in industrial CT scenarios.

### Strengths
The proposed Neural Adaptive Binning (NAB) method exhibits strong mathematical interpretability, which provides theoretical guarantees for rectangular priors.

### Weaknesses
The numerical results of the proposed method do not demonstrate irreplaceability, as comparable effects could be achieved through regularization techniques. Additionally, there is a lack of validation in specialized application scenarios.

### Questions
1. The authors mention that implicit neural representations with random Fourier coding often produce distinctive wave-like artifacts during reconstruction. However, upon comparing the INR results with the proposed method in Figures 5 and 6, the differences in reconstruction outcomes primarily manifest in the internal smoothness rather than at the edges. In practice, such artifacts can often be mitigated through regularization techniques. The authors are requested to clarify the advantages of their proposed method compared to alternative approaches, such as incorporating Total Variation (TV) regularization [1] into the network architecture.

[1] Deep Convolutional Neural Networks with Spatial Regularization, Volume and Star-Shape Priors for Image Segmentation. J. Math. Imaging Vis. 64(6): 625-645 (2022)

2. The results in Table 2 appear unusual, as the INR method shows better performance with 14 views than with 16 views. It seems counterintuitive and requires explanation.

3. The reconstruction results of INR in Figure 6 appear excessively smooth. Could this indicate potential overfitting to the training data?

4. The evaluation on medical datasets should be conducted to validate the robustness.

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
3

### Summary
The paper proposes a new encoding method used in implicit neural representation, which is learnable and can be used for image with simple structure. The experiments show the proposed method is effectively in sparse view CT reconstruction.

### Strengths
The self-supervised method can reconstruct the CT image with high quality.
Mathematical limits of the encoding result was given.

### Weaknesses
The proposed method has not evaluated with complex image. Such as medical image or industrial CT image with complex structure.

If the method can be adapted to 3D image reconstruction.

For the propose method needs about 30000 iterations. For the industrial scene，the time costs of proposed method is significantly. The comparison of time costs and iteration number with INR (Random Fourier Features) should be given.

Comparison with some other method is necessary, such as data-dirven tight frame method、DIP+TV method, also diffusion based SOTA method, DPS.

### Questions
What’s the performance of the proposed method with full view or less view (such as 8). 

How to select the hyper parameter M.

### Soundness
3

### Presentation
3

### Contribution
3

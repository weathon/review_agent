# NeuroSURF: Neural Uncertainty-aware Robust Surface Reconstruction

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 3, 5, 5

## Abstract
Neural implicit functions have become popular for representing surfaces because they offer an adaptive resolution and support arbitrary topologies. While previous works rely on ground truth point clouds, they often ignore the effect of input quality and sampling methods on the reconstruction. In this paper, we introduce NeuroSURF, which generates significantly improved qualitative and quantitative reconstructions driven by a novel sampling and interpolation technique. We show that employing a sampling technique that considers the geometric characteristics of inputs can enhance the training process. To this end, we introduce a strategy that efficiently computes differentiable geometric features, namely, mean curvatures, to augment the sampling phase during the training period. Moreover, we augment the neural implicit surface representation with uncertainty, which offers insights into the occupancy and reliability of the output signed distance value, thereby expanding representation capabilities into open surfaces. Finally, we demonstrate that NeuroSURF leads to state-of-the-art reconstructions on both synthetic and real-world data.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
NeuroSURF is a new method that takes posed depth images of an arbitrary static object as input and returns its SDF surface reconstruction. First, it computes mean and Gaussian curvatures of the backprojected depth in image space for each depth image. Then, the backprojected depth images/camera-space SDFs are merged in a coarse voxel grid. In addition, during merging, the curvatures are also merged and, as in a prior work on which the submission builds, each voxel also stores an uncertainty value and the gradient of the SDF. This voxel grid can then be very quickly queried to determine the curvature for arbitrary points via nearest-neighbor interpolation and uncertainty computation, which in turn allows to determine low-/middle-/high-curvature samples. Finally, these samples (positions, interpolated uncertainty, SDF gradient) are used to directly supervise a coordinate-based MLP that regresses SDF and uncertainty. I do not understand what happens afterwards. The main contribution/novelty lies in using curvature to guide the sampling and in using the uncertainty to mask out areas without supporting depth-image evidence during mesh extraction with Marching Cubes.

### Strengths
- The uncertainty value can be used as a mask when extracting a mesh from the SDF field via Marching Cubes. This enables open surfaces. 

- The scheme can be swapped into other methods, for example IGR, to improve their results. The experiments support this claim quantitatively. The authors promise to release code.

- The results on sparse (only a few thousand points, fairly uniformly distributed) are qualitatively and quantitatively much better than prior work.

### Weaknesses
*Method*

- Are depth discontinuities between neighboring pixels (e.g. from occlusions) taken into account when computing depth derivatives in image space? If not, that should be stated in the limitations.

- What is the point of transferring the information of the voxel grid into an MLP? The values extracted from the voxel grid are used directly for supervision of the MLP, such that the MLP will learn to simply reproduce the extracted values. Why not then directly treat the extracted values as the surface reconstruction functions? What does storing them in an MLP do? Memory savings? But that comes at the cost of speed, no?

- An overview figure would help with putting together the components. What is happening in Sec. 3.4? I think this part would benefit from feedback by someone not closely involved in the submission. It's currently plain confusing. Does SIREN/Neural-Pull become a part of NeuroSURF? If so, what does that mean? Or is the NeuroSURF sampling used in SIREN/Neural-Pull? I.e. instead of projecting a random point onto the closest surface by using the SDF and its gradient (obtained via back-prop through the SDF MLP of SIREN/Neural-Pull), the coarse voxel grid which stores SDF values and SDF gradients is queried? Doesn't turning that voxel grid into an MLP (see previous point) defeat the point then, which presumably is speed-up?

- Are Sec. 3.1 and 3.2 the core of the method, namely sampling; Sec. 3.3 is a straightforward application to surface reconstruction; and Sec. 3.4 is *another* application of Sec. 3.1 and 3.2 to surface reconstruction, by incorporating the sampling into prior reconstruction methods? Please add introductory sentences for context.

- Please explain better which parts of the method are responsible for which important downstream advantage. This motivation is missing. Why were these design choices made? What is the goal of each step of the method section?

*Results*

- There are a number of papers that I would like the authors to comment on in the rebuttal and in a revision of the submission, especially as to why comparisons to them are not required: 
* Duan et al. Curriculum DeepSDF discusses sampling/weighting schemes for training DeepSDF-like networks. 
* Hanocka et al. Point2Mesh is classical in spirit and allows for adding finer resolutions without re-running the entire coarser pipeline and it shows results on quite noisy/sparse point clouds, i.e. it does not share the limitations argued in Related Work regarding classical surface reconstruction methods. 
* Atzmon et al. SAL: Sign Agnostic Learning of Shapes from Raw Data is a deep implicit surface reconstruction method that works from unoriented and noisy raw point clouds, i.e. it does not share the limitations of learning-based methods mentioned in Related Work. 
* Takikawa et al. Neural Geometric Level of Detail is a major implicit surface reconstruction method. 
* Lindell et al. BACON: Band-limited Coordinate Networks for Multiscale Scene Representation is another major implicit surface reconstruction method.
* And a couple of papers from 2022 by Zhizhong Han beyond the 2020 paper Neural-Pull (which the submission compares to), namely Li et al. Learning Deep Implicit Functions for 3D Shapes with Dynamic Code Clouds and Ma et al. Reconstructing Surfaces for Sparse Point Clouds with On-Surface Priors.
* Also, why is there no comparison/ablation to Gradient-SDF, which appears to be the basis for the submission?

- Do SIREN, IGR and Ours use the same resolution and bounding box for Marching Cubes? IGR shows clear Marching Cubes artifacts, while the other two don't. That is relevant because IGR looks very similar to Ours except for these artifacts.

- As discussed in *Method* above, I'd like to see results when extracting values from the voxel grid directly (in the same manner as when generating samples for MLP supervision) instead of transferring them into an MLP. How does the quality change? What about memory and speed? 

- In the middle of the first paragraph of Sec. 4, it says that for methods from prior work that take point clouds as input, these point clouds are obtained via the voxel grid. Does that mean that the voxel grid from the submission is used for that? Wouldn't that imply that the voxel grid, i.e. the proposed method, upper-bounds all other methods because the other ones only get as input what the submission produces as intermediate output? That seems like an unfair setup?

- There are no qualitative results that allow to assess the qualitative difference that swapping the proposed method into IGR makes.

### Questions
As of now, the writing is too confusing and unclear. The motivation of design choices, context, and even the overall goal of the method is unclear to me. Furthermore, I'd like to see comparisons (or a discussion as to why that's unnecessary) to works from after 2020. These two aspects are my main objections to accepting the paper. 


*Minor notes*

Beyond the questions in Weaknesses, please address the following questions in a rebuttal:

- What is l_E, the last loss term in equation 9?

- Sec. 3.3 early on mentions an uncertainty threshold tau. What is it set to? 0?

- Is uncertainty ever used for anything other than as a binary mask during Marching Cubes where it is presumably "threshold-ed" at 0, i.e. parts that where never observed in the input depth images are removed?

- What is the point of the last paragraph of Sec. 3.4? Doesn't the ability to project points to the surface come from the prior work of Gradient-SDF? If so, it should be clearer that this paragraph isn't a contribution by the submission.

- Are arbitrary non-intersecting open surfaces possible? Using a mask in Marching Cubes seems like it might be restrictive? Beyond issues that would arise due to an insufficient resolution of the voxel grid?

- Are the three groups of curvature samples (end of Sec. 3.2) ever used for anything? Where in the method does it matter that they were split into three groups?

- Is there a reason why Gaussian curvature is never used in the experiments? A comparison between Gaussian and mean curvatures would be interesting for people who want to use the method. (This is optional since Gaussian curvature could also just be removed from the method section.)


These are just some other notes that do not need to be addressed:

- Not strictly necessary, but a point towards sophisticated classical shape representations would be helpful, e.g. Ohtake et al. Multi-level partition of unity implicits from 2005. 

- I'm not following how Sec. 3.1 and Sec. A.2.3 are connected. Does invariance to parametrization changes (main text) relate to the Jacobian of parametrization changes having non-zero determinant (appendix)? Please make the connection to the main text more explicit in the appendix.

- Should Equation 5 use psi_p instead of psi_x? psi_x is not defined as far as I can see.

- At the end of Sec. 3.2, there is a reference to Sec. 3.2.

- I assume Figures 11 and 12 belong to Sec. A.3? A sentence in Sec. A.3 referring to them would clarify that.

- Figure 9 does not look that convincing. The bunny evolves nicely during training both with and without the proposed sampling. Another shape might demonstrate the advantages better, like the right one. 

- The caption of Figure 8 should state what solid and dashed means, not just the main text on the previous page.

- Figure 5 could be easily extended by a neat ablation: a qualitative result from the full proposed method with and without the Marching Cubes thresholding.

### Soundness
3 good

### Presentation
1 poor

### Contribution
2 fair

---

## Human Reviewer 2

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
The manuscript introduces a pipeline designed for the reconstruction of surfaces from depth images. NeuroSURF is founded upon the voxelized SDF representation initially proposed by [Sommer et al, 2022]. It leverages a mean-curvature guided sampling approach, coupled with uncertainty values, to facilitate the extraction of open surfaces. NeuroSURF's performance was assessed across diverse datasets, encompassing synthetic scenes featuring ideal depth images, as well as real-world scans of indoor environments and objects.

### Strengths
-	The problem is of interest to the research community, as depth images can be cheaply available from depth sensors.
-	The formulation of open-surface extraction is intuitively sound, which has been a common issue for SDF representations.
-	The experiment setup is diverse, including real-world and synthetic datasets. This setup helps readers understand the values of the proposed approach.

### Weaknesses
- Theoretical Limitations:

  - **Curvature Sampling Limitations**: The utilization of curvature information to capture high-frequency surface details introduces certain limitations when employing mean and Gaussian curvatures. Sampling-based on mean curvature may overlook saddle regions where the curvature is positive in one direction but negative in another. Similarly, sampling based on Gaussian curvature might miss developable regions where one direction exhibits zero curvature while the other has high curvature. Although Figure 11 demonstrates the distinct biases of Gaussian and mean curvatures in sampling, the manuscript does not address their potential impact on performance.

  -  **Noise Sensitivity in Curvature-Based Sampling**: Sampling-based on curvature inherently exhibits sensitivity to noise. In the presence of a noisy depth map, all pixels tend to exhibit high curvature and, consequently, receive heavy sampling. The manuscript does not discuss strategies to mitigate this sensitivity to data noise effectively.

  - **Discontinuities in Voxelized Representation**:  The adoption of a voxelized representation with only a single center point and no interpolation between neighboring voxels can inherently introduce discontinuities in the SDF, as inferred from Equation 4 and the descriptions on Page 5. 



- Inadequate Experiment Results:
  - While NeuroSURF is compared against many recent generic representations such as SIREN and application-specific approaches such as Neural-Pull, the baselines in both cases are somewhat inadequate. 	
    - More recent generic representations such as Instant NGP [Müller et al, 2022] are not included, which potentially can resolve the lack of high frequency details issue. 
    - As for converting depth/point cloud to surfaces, there are also techniques such as recent NKSR [Huang et al, 2023], and differentiable possion SAP [Peng et al, 2021], which are robust to noise while recovering the geometric details.
    - The formulation (Equation 18-21) proposed in Appendix A.2.2 is similar to truncated signed distance function (TSDF) [Curless and Levoy, 1996]. In this case, I wonder if TSDF is adequate enough for recovering the surfaces accurately.

-	Unclear descriptions: 
  - The manuscript lacks a clear explanation of how uncertainties and SDF values are computed from the depth images. This information is crucial for understanding the constraints and potential limitations of the surface recovery process from depth data.
  - The use of notations within the manuscript can be confusing. It remains unclear what represents network estimates and what are the inputs, despite the presence of Table 4 in the appendix. Additionally, the manuscript does not provide formal definitions for certain notations like $\psi^x$ in Equation 5.


References: 

Müller, T., Evans, A., Schied, C., & Keller, A. (2022). Instant neural graphics primitives with a multiresolution hash encoding. ACM Transactions on Graphics (ToG), 41(4), 1-15.

Huang, J., Gojcic, Z., Atzmon, M., Litany, O., Fidler, S., & Williams, F. (2023). Neural Kernel Surface Reconstruction. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 4369-4379).

Curless, B., & Levoy, M. (1996, August). A volumetric method for building complex models from range images. In Proceedings of the 23rd annual conference on Computer graphics and interactive techniques (pp. 303-312).

Peng, S., Jiang, C., Liao, Y., Niemeyer, M., Pollefeys, M., & Geiger, A. (2021). Shape as points: A differentiable poisson solver. Advances in Neural Information Processing Systems, 34, 13032-13044.

### Questions
-	Given the limitations of mean/gaussian curvatures, could there be some ablations to further understand the impact on performance? Can the manuscript use total curvature instead? 
-	How sensitive the NeuroSURF is w.r.t noise? Could the manuscript provide some ablations?
-	Can the authors provide clarifications on if there are discontinuities across voxels?
-	Can the authors provide justifications for why some of the sensible baselines are not included? If they are indeed sensible baselines, could the manuscript include their results?
-	Could the authors provide clarifications on the preprocessing of depth images to obtain SDF and uncertainties? 
-	Could the manuscript clarify what are the optimizable variables and what are the inputs?

### Soundness
2 fair

### Presentation
2 fair

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
The paper introduces a technique for surface reconstruction from depth maps using an optimization-based approach. Central to this method is the emphasis on proficient sampling and filtering of input points, which are then integrated into a reconstruction method that employs an implicit field as its underlying representation. The filtering process proceeds by transforming the initial point cloud into voxels and adeptly determining attributes for these voxels, such as the SDF, gradient vector, and curvatures. Consequently, the optimization-based methods receive sampled points derived from the voxel grid structure. Beyond just surface attributes, the implicit field can incorporate surface uncertainty, facilitating intuitive open surface extraction and noise mitigation. When compared with several prevailing methods like Poisson Surface Reconstruction and IGR, the proposed sampling method's efficacy becomes evident.

### Strengths
- The technique addresses the innovative challenge of point cloud sampling specifically tailored for neural reconstruction from depth maps. It harnesses the voxel grid structure as a foundation to accelerate both computation and sampling processes.

- A distinctive feature of the method is its ability to produce uncertainty as an auxiliary output. This capability paves the way for a bunch of subsequent applications, including navigation.

- The experiments conducted are exhaustive, and the ablation analysis adeptly showcases the potency of each individual component.

### Weaknesses
- The motivation is somewhat unconvincing. The justification for adopting biased sampling remains ambiguous, and the choice to utilize voxel-based sub-samples over complete samples isn't adequately described.

- The section detailing the method is ambiguous, with many important details being omitted. This absence hinders the algorithm's reproducibility. Please see the 'Questions' section for a more in-depth breakdown.

- The computation of the uncertainty lacks a straightforward explanation. Specifically, where does the uncertainty come from? Are they coming from the quantization artifact introduced by voxelizing the points or coming from the sensor themselves (for example the angle between the surface normal and camera ray)? The authors should provide some intuitive examples demonstrating low and high confidence areas.

- The method's innovation is questionable. Employing curvature as a guiding principle for sampling isn't a novel approach (as already pointed out in the related works section). Moreover, the inherent characteristics of depth maps don't appear to be optimally leveraged.

### Questions
- How is the voxel grid built? How are the attributes such as gradient initialized for each voxel grid?

- Why is the geometry within each voxel being approximated as planar patches? Could fitting primitives such as ellipsoid or parabolic surfaces improve the results?

- Why is the uncertainty smaller if the points are far away from the voxel, as explained on Page 4?

- Fig.4 is not clear. Why would the sample still be evenly distributed on the surface given the curvature-aware sampling?

### Soundness
3 good

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 4

### Rating
5: marginally below the acceptance threshold

### Rating Number
5

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper addresses the problem of surface reconstruction from sparse input depth images/point cloud and is based on implicit neural shape representations. The key idea is a curvature guided sampling strategy that should help to improve reconstruction quality for sparse inputs by correcting for unevenly distributed points and enable interpolation among spares inputs. Furthermore the authors suggest integrating uncertainty in the loss and during the surface extraction which enables extracting open surfaces from implicit neural representations. The authors conduct experiments on object-level datasets as well as real-world datasets.The quantitative comparison on the object level shows especially good performance in the sparse inputs case.

### Strengths
The paper is well written and technical sound.
The curvature term is well explained and easy to follow how it is integrated in the method.
The idea of incorporating uncertainty to model open spaces with neural representations is novel and interesting.
Comparison with valid baselines is provided.

### Weaknesses
1) The loss function contains four parts. I’m missing an ablation study on the impact of each part for the final results, e.g. it stays unclear how much impact to the smoothness comes from the normal regularizer vs. from the interpolation sampling.
2) The paper does not discuss the network architecture of f(x, theta) in equation . As shown in previous works (SIREN, Fourier Features), there is a huge impact of the actual network architecture and input features to the final output shape, as the network acts as predefined prior. However, there is no explanation of the used architecture.

### Questions
It would be great to get more insights wrt. the loss function as mentioned in the weakness 1.

Regarding weakness 2) the role of f(x, theta) remains unclear to me. Please provide more explanation on that.

### Soundness
2 fair

### Presentation
3 good

### Contribution
3 good

# Falcon: Fast Proximal Linearization of Normalized Cuts for Unsupervised Image Segmentation

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 2, 6, 8

## Abstract
Current zero-shot unsupervised segmentation methods based on normalized cuts (NCut) face three key limitations. First, they rely on recursive bipartitions with repeated eigen-decompositions, making them prohibitively expensive at scale. Second, each split requires spectral relaxation followed by rounding, introducing layers of approximation where the final partition may diverge from the true NCut objective. Third, recursive bipartitioning offers no principled assurance of producing a stable $K$-way segmentation, and existing heuristics lack convergence guarantees. We propose \textbf{Falcon}, a proximal-gradient solver that directly optimizes the discrete $K$-way NCut objective without spectral relaxation. We prove linear convergence under the \textit{Kurdyka--\L{}ojasiewicz} (KL) property. Falcon computes closed-form gradient scores weighted by cluster volumes and performs row-wise one-hot proximal updates stabilized by inertia. A monotone backtracking scheme adaptively tunes the proximal parameter, ensuring non-decreasing NCut values. This design preserves discrete feasibility, removes repeated eigen-decomposition, and guarantees convergence. Across six benchmarks, Falcon outperforms the strongest official baseline (DiffCut) by wide margins, e.g., +13.2 mIoU on VOC, +27.7 on COCO-Object, and +3.1 on Cityscapes, while remaining competitive on Pascal Context. It also runs up to an order of magnitude faster than recursive NCut and scales more favorably in memory at high resolution, making it practical for larger token grids. By pairing pretrained foundation models with a principled NCut solver, Falcon sets a new state of the art across six benchmarks and achieves the best performance on 17 of 18 benchmark--encoder pairs, underscoring both its robustness and its generality in bridging the gap between unsupervised and supervised segmentation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper proposes “Falcon,” a graph cut algorithm for performing unsupervised image segmentation using embeddings from pretrained vision transformers. This algorithm is a proximal-gradient solver that directly optimizes the discrete K-way NCut objective without spectral relaxation. The paper shows this proposed method outperforms baseline approaches on common datasets and provides ablation and runtime experiments.

### Strengths
Technical novelty and practical impact
- The authors propose a new graph cut method that is both performant and efficient for solving unsupervised segmentation problems provided a pretrained vision transformer.
- The authors show improved performance compared to baseline methods (MaskCLIP, MaskCut, DiffSeg, DiffCut, AutoSC) on common benchmark datasets (VOC, Context, COCO-Object, COCO-Stuff-27, Cityscapes, ADE20K), suggesting the proposed method improves unsupervised segmentation with pretrained transformer models. 

Helpful ablation experiments
- The authors perform helpful ablations showing the method improves performance over multiple pretrained networks and evaluating the refiner on both the proposed method as well as with baseline methods. 
- The authors provide detailed runtime experiments showing the method improves inference speed compared to two baselines.

### Weaknesses
Baseline method selection
- The authors focus on a few previously published methods and show the proposed method convincingly outperforms them, but do not convince the reader these are representative/comprehensive baselines. Authors should clarify how and why they choose what baseline to include vs. exclude (e.g., zero shot SAM; STEGO; diffuse attend and segment; U2Seg; etc.), and consider including additional baselines if there is no convincing reason they should be excluded.

Reproducibility
- Will the authors publish code to make this method usable/reproducible? If not, I have doubts about the reproducibility of this method.

There are a few items that need to be clarified to contextualize the strengths of the method:
- Why are images resized to 1024 x 1024? This seems higher resolution than is typical for these natural image datasets and would increase runtime with the pretrained transformers. It also seems like this resize may be inflating inference speedups.
- How is K determined? You say “K is determined via spectral thresholding: we count eigenmodes below a negative threshold κ and clamp K to a predefined range, “ but it is unclear to me what this means. Is K a range, and final K value is selected via best empirical performance? Does this need to be hand tuned per dataset? Where in the method pipeline is this operation performed? What is the rationale for the negative threshold and clamping? Is kappa hand tuned per dataset?
- Can the authors comment on how memory usage compares to baseline methods? Since many of the intermediate matrices are NxN, is there a limit on image/token size * feature size (both dependent on the pretrained network) that is practical with this method?
- Since gradient scores are weighted by cluster volumes, how do the authors expect volume size to impact performance? Is this dependence seen in the empirical results? (e.g., significantly stronger performance on large objects vs. small)

Minor weaknesses:
- I think the authors have used \cite{}, which includes citations in-text, where they should have used \citep{}, which includes citations in parentheses. The authors should switch the citations to the correct format as appropriate to improve readability. 
- Typo on line 43, “as” instead of “has.”
- The authors lay out 3 challenges with existing segmentation methods in the abstract. Similarly, they list 3 challenges with existing segmentation methods in the introduction. However, these 3 challenges are not the same in the abstract vs. intro. This makes it challenging as a reader to know what limitations with previous methods the authors are aiming to address. The authors should make the abstract and intro consistent with one another.
- Line 242, “(iii)” is repeated twice.
- The authors lay out issue with “classical refiners” in lines 295-297, but do not state how PAMR or NAMR get around these challenges. A transition paragraph would help. 
- (H, W) seems to be overloaded notation. The authors use (H,W) to describe the size of the intermediate mask/features in Section 3.3, but then use (H,W) to describe the full resolution image in Section 3.4. Are (H,W) always the same, i.e. is the intermediate mask in Section 3.3 always at full resolution? If so, that is not clear from Section 3.3, as you refer to this stage as “intermediate-resolution refinement.”
- There are no segmentation visualizations in the main text, they are instead put in the supplementary. If the authors have space, consider moving these to the main text. Additionally consider including visualizations of baseline vs. Falcon results for the reader to qualitatively understand differences in the methods.
- There are many parameters given in Section 3; authors should include defaults for each of these in either the main text or supplementary.

### Questions
Addressed in Weaknesses section.

### Soundness
3

### Presentation
2

### Contribution
3

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper presents a method for unsupervised semantic image segmentation. The begins by image feature extraction using pre-trained models such as Vision Transformer trained with DINO. An affinity matrix is assembled by pairwise feature similarity. The affinity matrix elements are viewed as weights on graph edges. This enables a bipartitioning objective, where the resulting partitions equate to segments in the input image. This paper presents Falcon, an extension of Normalized Cuts and recently proposed Mask Cut algorithms that partition the graph in a recursive fashion; more than $K=2$ partitions are obtained by recursively partitioning the remaining subgraph. Instead, Falcon partitions into $K>2$ groups at once and avoids eigenvalue decomposition, reducing the method complexity from $\mathcal{O}(N^3)$ to $\mathcal{O}(KN^2)$. Since tokens are subsampled wrt. the input image resolution, the authors propose two methods to recover partitioning at a higher resolution.

### Strengths
This paper considers an important topic in computer vision. Contributions to semantic image segmentation are meaningful and relevant. The work is also well motivated; current methods based on graph partitioning depend on eigenvalue decomposition that is $\mathcal{O}(N^3)$ in time, making them unscalable as input resolution grows. The experimental section includes evaluations on six datasets.

### Weaknesses
The method section is really complex and difficult to follow. There is an abundance of symbols that are not gradually included, and the text between the equations does not connect the formalism in understandable fashion. For example, most quantities are not expressed in text: readers would appreciate understanding what do $\mathbf{v}$ and $\mathbf{q}$ hold, and that $v_k$ and $c_k$ are merely elements within these vectors. Some variables are multiplied. For instance, a square similarity matrix $\mathbf{S}$ is introduced in Equation (2). Afterwards, a $N \times K$ matrix $\mathbf{S}$ is used in Equation (8). Similarly, $c_k$ (Equation(9)) and $\mathbf{c}_k$ (Equation(13)) refer to different quantities. Equation (11) expresses the update in the proposed iterative algorithm, but it seems to render improperly in the PDF.

The experimental section lacks proper comparison with related work.
Firstly, influential relevant related methods, U2Seg [1] and UnSAM [2], are not included in experimental comparison.
Secondly, since CutLer uses MaskCut to generate instance segmentation pseudo labels, it is unclear how is Mask Cut adapted for unsupervised semantic segmentation. This is important to note as Falcon is most related to Mask Cut. Moreover, the original work uses DINOv1 and CRF [3] to generate instance pseudo labels. A fair comparison should include the same backbone and postprocessing to highlight the contribution. Furthermore, the proposed postprocessing methods are not validated against the commonly used CRF postprocessing, therefore there is no validation of the proposed postprocessing methods.

Execution time measurements do not include a comparison with Mask Cut. Also, separate time measurements for graph partitioning and postprocessing (the two contributions) are missing.
The authors comment that CRF post processing is CPU-bound. This is true for the official Mask Cut implementation [4], but does hold in general [5]. To highlight the lower theoretical complexity of Falcon, the experimental section could include measurements under common implementation and hardware*.

- [1] Niu, Dantong, et al. "Unsupervised universal image segmentation." Proceedings of the IEEE/CVF conference on computer vision and pattern recognition. 2024.
- [2] Wang, XuDong, Jingfeng Yang, and Trevor Darrell. "Segment anything without supervision." Advances in Neural Information Processing Systems 37 (2024): 138731-138755.
- [3] Krähenbühl, Philipp, and Vladlen Koltun. "Efficient inference in fully connected crfs with gaussian edge potentials." Advances in neural information processing systems 24 (2011).
- [4] https://github.com/facebookresearch/CutLER/blob/main/maskcut/maskcut.py
- [5] https://github.com/heiwang1997/DenseCRF

$*$ Note that Mask Cut performs eigenvalue decomposition on CPU.

### Questions
- The number of clusters $K$ is "is determined via spectral thresholding". What is the complexity of this algorithm?
- How does the accuracy depend on the number of iterations $\mathbf{X}^{(t)}$? How many iterations does Falcon use in practice?
- How many iterations are required to perform the line search in each iteration from Equation (12)?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces FALCON, a proximal-gradient solver for the discrete K-way normalized-cut (NCut) problem in unsupervised image segmentation. FALCON directly optimizes discrete assignments without spectral relaxation or eigen-decomposition. Experiments show the performance on six segmentation benchmarks (VOC, COCO-Object, Cityscapes, etc.)

### Strengths
1. The paper reformulates the classical NCut objective as a proximal-gradient problem with discrete feasibility preserved at every step, eliminating spectral relaxation and rounding. The idea is sound and interesting.
2. Experiments across six public benchmarks validate the performance of the proposed method.

### Weaknesses
1. It seems that the proposed method is mathematical. But the the intuition for why the proximal scheme works well is not clear.
2. The method relies on pretrained vision encoders (e.g., SSD-1B, DINOv3), raising questions about its standalone generality in low-resource or domain-shift settings.

### Questions
1. Sensitivity of hyperparameters. How sensitive is convergence to the choice of the backtracking parameter $\gamma$ and the inertia term?
2. Can FALCON handle dynamic or streaming data scenarios where affinities evolve online?
3. The number of segments $K$ is determined by counting eigenmodes below a cutoff $k$. How robust is this spectral thresholding across datasets, and could a data-adaptive or learnable k further improve stability?

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper presents a new pseudo-mask generation framework, Falcon, which incorporates fast proximal linearization and adaptive mask refinement to achieve better or the same unsupervised segmentation performance with much faster generation.

### Strengths
1. good motivation. Repeated NCut is slow. This paper presents a faster generation method with better performance.
2. clear formulation and presentation. This paper formulates the generation problem using many formulas and proofs.
3. efficiency analysis. Inference time of different methods are compared.
4. encoder ablation. The author presents ablation of different SSL encoder to validates the effectiveness of Falcon.

### Weaknesses
1. While Falcon is more efficient than recursive NCut methods, its complexity still scales quadratically with the number of tokens, making it less optimal for extremely large datasets or scenarios with very high token resolutions.

### Questions
None

### Soundness
3

### Presentation
4

### Contribution
4

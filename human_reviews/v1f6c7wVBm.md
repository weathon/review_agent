# AniSDF: Fused-Granularity Neural Surfaces with Anisotropic Encoding for High-Fidelity 3D Reconstruction

- Decision: Accept (Poster)
- Scores: 6, 6, 8, 6, 6

## Abstract
Neural radiance fields have recently revolutionized novel-view synthesis and achieved high-fidelity renderings. 
However, these methods sacrifice the geometry for the rendering quality, limiting their further applications including relighting and deformation. 
How to synthesize photo-realistic rendering while reconstructing accurate geometry remains an unsolved problem. In this work, we present AniSDF, a novel approach that learns fused-granularity neural surfaces with physics-based encoding for high-fidelity 3D reconstruction. Different from previous neural surfaces, our fused-granularity geometry structure balances the overall structures and fine geometric details, producing accurate geometry reconstruction. 
To disambiguate geometry from reflective appearance, we introduce blended radiance fields to model diffuse and specularity following the anisotropic spherical Gaussian encoding, a physics-based rendering pipeline. With these designs, AniSDF can reconstruct objects with complex structures and produce high-quality renderings. 
Furthermore, our method is a unified model that does not require complex hyperparameter tuning for specific objects. 
Extensive experiments demonstrate that our method boosts the quality of SDF-based methods by a great scale in both geometry reconstruction and novel-view synthesis.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The AniSDF paper introduces an innovative approach to high-fidelity surface reconstruction and photo-realistic rendering from multi-view images. This is achieved through a synergistic geometry network and appearance network, which together enable high-quality 3D reconstruction. Additionally, the authors propose a fused-granularity neural surface that aims to balance overall structural integrity with fine detail preservation.

### Strengths
Pros:
The paper is well-written and generally easy to follow.
Experimental results demonstrate incremental improvements in PSNR, which support the proposed approach.

### Weaknesses
Cons:
The fused-granularity neural surface structure may lack novelty, as it essentially uses two parallel structures with different resolutions. It seems likely that resolution choices could impact the final reconstruction quality. Including experiments that vary resolution settings would clarify their effect. 


Despite claims of high-quality mesh reconstruction, Chamfer Distance results reveal performance gaps on certain objects (e.g., "Chair" and "Mic" categories) compared to methods like Neus and NeRO. Explaining these discrepancies would help elucidate the limitations.

### Questions
It’s unclear whether the larger network or the fused-granularity neural surface structure is responsible. What would happen if we set both the fine and coarse grids to the same resolution, either that of the coarse grid or that of the fine grid?

I raise my ratings. Good Luck.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
The paper presents AniSDF, a novel SDF-based method for high-fidelity 3D reconstruction that incorporates fused-granularity neural surfaces and anisotropic spherical Gaussian (ASG) encoding. AniSDF aims to achieve accurate geometry reconstruction and photo-realistic rendering by addressing challenges in neural radiance fields, such as geometry-appearance trade-offs. The approach uses parallel fused-granularity neural surfaces to balance coarse and fine details, and blended radiance fields with ASG encoding for modeling both diffuse and specular appearances. Extensive experiments demonstrate AniSDF's superiority in both geometry reconstruction and novel-view synthesis over prior methods.

### Strengths
1.	Innovative Approach: The use of fused-granularity neural surfaces combined with ASG encoding for 3D reconstruction is novel and effective in balancing both coarse and fine details, leading to improved geometry and appearance quality.
2.	High Performance: Experimental results show significant improvements in rendering quality and geometry reconstruction, with better handling of reflective, luminous, and fuzzy objects compared to existing methods.

### Weaknesses
1.	Limited Real-Time Capability: AniSDF cannot perform real-time rendering, which limits its applicability in time-sensitive applications such as interactive graphics or augmented reality.
2.	Computation Cost: The use of multiple neural networks and high-resolution hash grids could be computationally expensive, which may hinder scalability.

### Questions
Questions:
1. I compared the chamfer distance metric of the Neuralangelo method reproduced on the DTU dataset in the paper, and there is a significant gap. The original paper reported an average of 0.61 (which surpasses your method), while the reproduced result in the paper is 1.07. Could the authors clarify the reasons for this discrepancy? Specifically, did you maintain the same hyperparameter settings as Neuralangelo? Please provide detailed information on your experimental setup.
2. Section 3.2 of the paper lists some issues related to coarse grid and fine grid training, but there are no corresponding experimental supports for these claims. Regarding the use of the coarse to fine method, you pointed out that thin structures may be discarded in the early training stages. Could you provide visualizations of surface reconstruction at different training stages, along with corresponding quantitative metrics, particularly for Neuralangelo and Neus2? This would help us assess the advantages and disadvantages in the reconstruction of detailed structures. I noticed your experiments in the ablation study, but they do not specify the experimental settings and only show the final results.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
This paper introduces AniSDF, an approach to enhance the quality of SDF-based methods in geometry reconstruction and novel-view synthesis tasks, enabling the physically-based rendering ability. Firstly, AniSDF uses the parallel branch structure of coarse hash-grids and fine hash-grids, replacing the former sequential coarse-to-fine training strategy, to learn a fused-granularity neural surface to improve the quality of SDF. Secondly, AniSDF uses Anisotropic Spherical Gaussian Encoding to learn blended radiance fields with a physics-based rendering, disambiguating the reflective appearance.

### Strengths
Originality:
The paper explores the potential of parallel using coarse and fine hash-grid to replace the general sequential coarse-to-fine structure, demonstrating the effects of experiments. Besides, this paper combines SDF learning with blended radiance field learning with anisotropic spherical Gaussian encoding to distinguish material information.

Quality:
The quality of the paper is good, evidenced by detailed experiments and comprehensive comparisons with state-of-the-art methods. 

Clarity:
The paper is well-structured and organized. 

Significance:
The good geometry that disambiguates the reflective appearance is helpful in 3D reconstruction. The possible relighting application makes this research meaningful.

### Weaknesses
1. Reference is insufficient: From Lines 130 to 135, the Sec. 2.1 is related to the attempts to improve the reconstructed geometry of Gaussians. Since 2DGS is used to compare, it is no reason to cite some papers focused on doing similar jobs: improving the surface reconstruction of 3DGS, like $\cite{guedon2023sugar, lyu20243dgsr, chen2023neusg}$.

2. The ablation study of the fused-granularity neural surface is not enough. The ablation study shows the comparison with the sequential coarse-to-fine method, but the technique with only coarse hash-grid and only fine hash-grid should also be demonstrated to prove the observations shown at the beginning of Sec.3.2. It could be better if the training time comparison is also shown in this part.

@article{guedon2023sugar,
  title={SuGaR: Surface-Aligned Gaussian Splatting for Efficient 3D Mesh Reconstruction and High-Quality Mesh Rendering},
  author={Gu{\'e}don, Antoine and Lepetit, Vincent},
  journal={CVPR},
  year={2024}
}
@article{lyu20243dgsr,
  title   = {3DGSR: Implicit Surface Reconstruction with 3D Gaussian Splatting},
  author  = {Xiaoyang Lyu and Yang-Tian Sun and Yi-Hua Huang and Xiuzhe Wu and Ziyi Yang and Yilun Chen and Jiangmiao Pang and Xiaojuan Qi},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2404.00409}
}
@article{chen2023neusg,
  title   = {NeuSG: Neural Implicit Surface Reconstruction with 3D Gaussian Splatting Guidance},
  author  = {Hanlin Chen and Chen Li and Gim Hee Lee},
  year    = {2023},
  journal = {arXiv preprint arXiv: 2312.00846}
}

### Questions
1. Modify the typo in Line 093 (‘Oue’) to ‘Our.’

2. The physical-based rendering method via ASG is similar to $\cite{yang2024spec}$. Is this work also inspired by similar works using ASG to learn the specular representation in 3DGS?

3. The blended radiance fields with ASG encoding are composed of $c_{view}$ and $c_{ref}$ though a learnable weight. According to Eq. 4, the light field is modeled by $c_d$ and $c_s$, diffuse color and specular color. So, can $c_{view}$ be regarded as purely diffuse and $c_{ref}$ as the pure composition of specular? If so, can the radiance fields be considered purely diffuse when $\omega$ is 1? If not, can this work disentangle the light field to only diffuse or specular field?

4. Refer to $\cite{han2023multiscale}$, they control the final color by adding the scale to the color calculated from ASG when retaining the diffuse color term calculated from the first three orders of SH. So, what motivates adding weight to diffuse and specular in this work? What are the differences between your light field calculation in the blended radiance fields with $\cite{han2023multiscale}$?

@article{han2023multiscale,
  title     = {Multiscale Tensor Decomposition and Rendering Equation Encoding for View Synthesis},
  author    = {Kang Han and Weikang Xiang},
  journal   = {Computer Vision and Pattern Recognition},
  year      = {2023},
  doi       = {10.1109/CVPR52729.2023.00412},
  bibSource = {Semantic Scholar https://www.semanticscholar.org/paper/aa41843888fffada6335b6c5cdbcd2d4bb5cf9da}
}

@article{yang2024spec,
  title={Spec-gaussian: Anisotropic view-dependent appearance for 3d gaussian splatting},
  author={Yang, Ziyi and Gao, Xinyu and Sun, Yangtian and Huang, Yihua and Lyu, Xiaoyang and Zhou, Wen and Jiao, Shaohui and Qi, Xiaojuan and Jin, Xiaogang},
  journal={arXiv preprint arXiv:2402.15870},
  year={2024}
}

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
4

### Summary
This paper provides a fused-granularity neural surfaces with physics-based anisotropic
spherical Gaussian encoding for high-fidelity 3D reconstruction. The authors show state-of-the-art novel-view rendering and geometry reconstruction results on several datasets, including NeRF-Synthetic, Shiny Blender, and DTU datasets. The proposed method shows very convincing reconstruction of challenging specular and furry objects.

### Strengths
1. This paper is well-written and easy to follow. 
2. Qualitative and quantitative results of the proposed method seems very strong, beating prior baselines like NeuS, RefNeRF, RefNeuS. Reconstructed meshes look clean with high-quality surface normals. 
3. The authors validate the proposed method on both synthetic datasets (Nerf-synthetic and Shiny-blender), and real datasets (DTU), and it's nice to see improvements.

### Weaknesses
1. To further convince me about the method's performance on reconstructing specularity, I would need to see the view synthesis videos, as opposed to the static frames shown in the paper and project page. Unfortunately, I could not find such videos (except the relighting ones). 

2. For the proposed blended radiance field (Eq. 12), I think it would be great to provide some visualizations of the individual components: w, c_view, c_ref, to better understand what each learnt components look like. 

3. I'm unsure why the fused-granularity hash grids actually works better than a plain multi-resolution hash grids. It seems to me that the major difference of it from a plain one is the additional handcrafted equation (6) that says the final SDF is an addition of coarse and fine SDF. In the plain multi-resolution hash grid, the final SDF is predicted by a MLP from concatenated multi-resolution features. This could benefit some justification.

### Questions
1. For Eq. 10, why would both c_view and c_ref depend on view directions? Would it encourage better diffuse-specular separation if one makes c_view view-independent?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 5

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper presents a high-quality 3D surface reconstruction method- AniSDF, which learns fused-granularity neural surfaces with physics-based encoding. The authors propose fused multi-resolution grids for geometry modeling, and adopt Anisotropic Gaussians for appearance modeling. With these designs, AniSDF can reconstruct objects with complex structures and produce high-quality renderings on benchmarked datasets.

### Strengths
1. The paper is well written and easy to follow. The idea is clean and the pipeline do not introduce additional hyper-parameter tuning and selection compared to some other recent methods for neural surface reconstruction / rendering.
2. The idea to fuse multi-resolution grids for detailed surface reconstruction is novel. AniSDF’s fused-granularity structure balances high- and low-resolution information to improve convergence and accuracy. This approach allows for a more adaptive reconstruction that captures both overall structure and fine details, which is validated by their good geometry quality (chamfer) in the experiments.
3. The use of ASG encoding in appearance modeling seems to be effective and handles specular reflections very well.

### Weaknesses
1. Experiments for reflective surfaces mainly come from synthetic data, it would be helpful to understand the model’s ability if we could see results of more real-world reflective surface data, such as trucks from Tanks and Templates, sedans from Refnerf, The Glossy-Real dataset from Nero.
2. The appearance modeling involves blending view-based and reflection-based radiance fields, however, the method's ability to decompose base color and reflection color is unknown.  It would be better if the author could add a visualization of view-based color, reflection-based color, and the blending weight.
3. Some details of the methods are missing. The derivation of normal and the method for mesh extraction are not discussed.
4. The reason behind the choice of grid level `m` and `l`  is not clear. It would be clearer if an ablation study about grid resolution were added.

### Questions
1. Add real-world experiments and more benchmark datasets (with larger scene scale).
2. Add visualizations to the decomposed appearance throughout the figures in the paper.
3. Add missing details in the method section.
4. Add some comparison on the train - test efficiency and memory footprint.
4. Revised the discussions and ablation studies as suggested in the weaknesses.

### Soundness
2

### Presentation
3

### Contribution
3

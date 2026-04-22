# Dynamic Novel View Synthesis in High Dynamic Range

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 8, 4, 6, 4

## Abstract
High Dynamic Range Novel View Synthesis (HDR NVS) seeks to learn an HDR 3D model from Low Dynamic Range (LDR) training images captured under conventional imaging conditions. Current methods primarily focus on static scenes, implicitly assuming all scene elements remain stationary and non-living. However, real-world scenarios frequently feature dynamic elements, such as moving objects, varying lighting conditions, and other temporal events, thereby presenting a significantly more challenging scenario. 
To address this gap, we propose a more realistic problem named HDR Dynamic Novel View Synthesis (HDR DNVS), where the additional dimension ``Dynamic'' emphasizes the necessity of jointly modeling temporal radiance variations alongside sophisticated 3D translation between LDR and HDR. To tackle this complex, intertwined challenge, we introduce HDR-4DGS, a Gaussian Splatting-based architecture featured with an innovative dynamic tone-mapping module that explicitly connects HDR and LDR domains, maintaining temporal radiance coherence by dynamically adapting tone-mapping functions according to the evolving radiance distributions across the temporal dimension. As a result, HDR-4DGS achieves both temporal radiance consistency and spatially accurate color translation, enabling photorealistic HDR renderings from arbitrary viewpoints and time instances.
Extensive experiments demonstrate that HDR-4DGS surpasses existing state-of-the-art methods in both quantitative performance and visual fidelity. Source code is available at \url{https://github.com/prinasi/HDR-4DGS}.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
5

### Summary
This paper studies a new problem – high dynamic range dynamic novel view synthesis (HDR DNVS). First of all, the authors collect a 4D synthetic datasets with 8 scenes and a 4D real datasets with 4 scenes. On the set up benchmark, the authors develop a HDR-4DGS, featuring dynamic tone-mapping for adaptively bridging the LDR and HDR domains under complex spatiotemporal variations.

### Strengths
(i) This paper studies a brand new problem - high dynamic range dynamic novel view synthesis (HDR DNVS), which has not been explore in the literature. This is novel and interesting. To this end, the authors made a dataset contribution – collecting both synthetic and real datasets to support the research. This is solid, non-trivial, and labor-intensive.

(ii) The proposed biologically inspired dynamic tone-mapping module also shows deep insights. This module draws inspiration from human visual adaptation, where retinal photoreceptors dynamically adjust to ambient brightness. The proposed HDR-4DGS includes a dynamic radiance context learner that models temporal radiance distributions, as shown in Figure 1.

(iii) The writing is good and easy to follow. Especially the method part, all the technical details are very clear. The presentation is also well-dressed. For example, the workflow of the overall pipeline can be clearly shown in Figure 1. In addition, the arrangement of the paper's figures and tables is very neat such as Tables 1, 2 and Figures 2, 3.

(iv) The performance is very solid. As compared in Tables 1 and 2, the proposed HDR-4DGS achieves much better reconstruction quality while keeping a very fast speed. The visual comparisons in figures 2 and 3 also suggest that the proposed HDR-4DGS can render more visually pleasant results with less blur and artifacts.

(v) The ablation study is also very comprehensive. The effects of dynamic tone mapper, dynamic radiance context learner, pixel-level supervision, DRCL design, and the temporal context length are discussed in details.

### Weaknesses
(i) Although the authors claim the datasets and the new algorithms are their contributions, there are no datasets or codes submitted. The reproducibility cannot be checked.

(ii) The authors claim the dataset is also a contribution. But there are very few words in the paper introduce how to construct the datasets. It would better if there is a section to describe the details of data collection.

(iii) As the authors claim, the designed method is a 4D Gaussian Splatting. But I did not see any preliminary or method part introduce how to adapt 4DGS to the HDR-DNVS task.

### Questions
What is the difference between the designed method with HDR-GS + the k-planes in 4DGS? How about the performance of HDR-GS + the k-planes in 4DGS.

### Soundness
4

### Presentation
3

### Contribution
4

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
5

### Summary
This paper focuses on learning HDR 3D representations of dynamic scenes from LDR images. Unlike previous methods that are limited to static scenes, this work introduces a dynamic tone-mapping module to achieve adaptive color transformation. By integrating this module with the Gaussian splatting representation, the proposed approach enables HDR rendering from arbitrary viewpoints and time instances. The method is evaluated on both real and synthetic datasets, with some experiments confirming its effectiveness.

### Strengths
- The paper extends HDR modeling from static to dynamic scenes by proposing a novel HDR-4DGS framework.
- The paper introduces a dynamic tone-mapping module to handle color translation in dynamic scenes, enabling the rendering of HDR images.
- The method is evaluated on both synthetic and real datasets, and the qualitative and quantitative experiments shown in the paper suggest that it is partially effective in performing color transformations.

### Weaknesses
This work proposes a dynamic HDR scene modeling framework that extends Gaussian Splatting to handle dynamic scenes. While the method achieves improvements on part of the experimental data, it still exhibits several concerns, as summarized below：

- Limited performance. As shown in the local results in Figures 2 and 3, the proposed method struggles to recover fine-texture details, and the synthesized novel views exhibit noticeable noise. The authors are encouraged to provide a more detailed analysis of these experimental results to support the effectiveness of the proposed approach better. Furthermore, based on the demo results, the method appears unable to preserve the geometry of moving objects in real-world scenes, often leading to structural loss during deformation. In addition, the DVS results show temporal flickering artifacts, indicating instability in dynamic rendering.

- Limited novelty. According to the paper, the dynamic scene modeling part is adopted from 4DGS, while the per-channel tone-mapping functions adopt the structure from HDR-GS. Thus, the contribution primarily lies in combining these existing components with an additional temporal HDR learning, resulting in an improvement over prior methods that is relatively incremental.

- Insufficient constraints of dynamic Gaussian deformation. As shown in Section 3.4, the method primarily supervises and optimizes the mapping between LDR and HDR domains. Still, it lacks explicit mechanisms or loss functions to control or regularize the deformation of dynamic Gaussians, which are essential for preserving the geometry and spatiotemporal consistency of moving objects. However, such constraints are crucial for dynamic scene representation and novel-view synthesis, and this aspect appears to be missing in this paper.

- Some comparative experiments are missing. Since the proposed method targets dynamic scenes, whereas the comparison methods in the paper mainly focus on demonstrating HDR visual quality, the authors could first map LDR images to HDR and then compare them with dynamic NeRF and Gaussian Splatting approaches to evaluate performance in terms of geometric accuracy and spatiotemporal consistency for moving objects.

### Questions
Based on the aforementioned weaknesses, the following issues also require discussion:

- In real-world scenes, as shown in Fig. 2, when using both LDR and HDR images as supervision for HDR rendering, the proposed method performs significantly worse than HDR-HexPlane in quantitative metrics. The authors should provide a clear and thorough analysis to explain this performance gap.

- The paper reports strong results on synthetic data but only limited improvement on real-world datasets. Can you elaborate on whether this discrepancy arises from domain differences, noise in HDR ground truth, or instability in the tone-mapping process?

- The authors could consider providing the output luminance as a function of input intensity for different time frames or varying brightness levels (i.e., a dynamic CRF). This would provide more convincing evidence that the proposed DTM has indeed learned dynamic mapping patterns, rather than relying solely on final HDR rendering metrics.

[Minor]:
- In Figure 1, the notation for the HDR and LDR models appears to be identical, which may cause confusion regarding their respective roles in the framework. And the equations shown in this figure should be formatted consistently with those in the manuscript to ensure clarity and uniform presentation.

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper addresses the limitations of existing High Dynamic Range Novel View Synthesis (HDR NVS) methods, which are confined to static scenes, and Dynamic Novel View Synthesis (DNVS) approaches that lack HDR support. It formally introduces the task of HDR Dynamic Novel View Synthesis (HDR DNVS) for the first time.

To tackle the core challenges of spatiotemporal radiance consistency and LDR-HDR domain translation in this task, the authors propose the HDR-4DGS framework. Built on 4D Gaussian Splatting, this framework innovatively integrates a Dynamic Tone-Mapping Module (DTM). The DTM stores temporal radiance statistics via a radiance bank, captures temporal variations using a dynamic radiance context learner, and drives per-channel tone-mapping functions to enable adaptive domain translation.

Additionally, the paper constructs two benchmark datasets: HDR-4D-Syn (8 synthetic scenes) and HDR-4D-Real (4 real-world scenes), which provide HDR ground truth and multi-view LDR observations. Experiments demonstrate that HDR-4DGS outperforms existing methods in both quantitative metrics (e.g., PSNR, SSIM) and visual quality, while balancing training efficiency and inference speed, offering an effective solution for high-fidelity HDR novel view synthesis of dynamic scenes.

### Strengths
1. The proposed task of HDR Dynamic Novel View Synthesis (HDR DNVS) is a novel and necessary one. Existing methods either focus on static scenes for HDR Novel View Synthesis (HDR NVS) or are limited to Low Dynamic Range (LDR) inputs for Dynamic Novel View Synthesis (DNVS), failing to handle real-world dynamic scenarios with time-varying geometry, illumination, and high-contrast radiance. This gap significantly restricts practical applicability, making the introduction of HDR DNVS a timely and problem-solving initiative.

2. The newly proposed Dynamic Tone-Mapping (DTM) module shows considerable merit, with a clear and logically coherent presentation in the paper. By constructing a radiance bank to store temporal radiance statistics, integrating a dynamic radiance context learner to model temporal variations, and designing per-channel tone-mapping functions, the module explicitly bridges the HDR-LDR domains while maintaining spatiotemporal radiance consistency. Its design draws reasonable inspiration from human visual adaptation, and the technical details (e.g., sliding window for context extraction, joint use of exposure time) are well-explained, making the module’s working mechanism easy to follow.

3. The experimental design is comprehensive and rigorous. First, the authors build two benchmark datasets (HDR-4D-Syn with 8 synthetic scenes and HDR-4D-Real with 4 real-world scenes) that provide essential HDR ground truth and multi-view LDR data for evaluating HDR DNVS, filling the lack of existing benchmarks. Second, they compare HDR-4DGS with multiple state-of-the-art methods (e.g., HDR-NeRF, HDR-GS, HDR-HexPlane) across quantitative metrics (PSNR, SSIM, LPIPS) and qualitative visualizations. Additionally, ablation studies verify the effectiveness of the DTM module and key components (e.g., pixel-level supervision, GRU-based context learner), ensuring the reliability of the results

### Weaknesses
1.The scale of the constructed datasets appears relatively limited, which may restrict further in-depth analysis and generalization verification. Specifically, HDR-4D-Syn only includes 8 synthetic scenes and HDR-4D-Real has 4 real-world indoor scenes, with a lack of diversity in scene types (e.g., no outdoor dynamic scenes with complex natural lighting changes, or scenes with more diverse object motions). Such a small dataset not only makes it difficult to fully validate the model’s robustness across different dynamic scenarios but also cannot support additional explorations (e.g., analyzing the model’s performance under varying motion intensities or lighting contrasts). This limitation may weaken the persuasiveness of the model’s practical applicability in broader real-world contexts.

2.The proposed Dynamic Tone-Mapping (DTM) module, though conceptually reasonable, seems relatively general as it lacks testing on the dataset associated with HDR-GS (a representative method for static HDR Novel View Synthesis). HDR-GS is a key baseline in the HDR NVS field, and its dataset likely contains rich HDR-LDR paired data suitable for evaluating tone-mapping performance. Without testing the DTM module on HDR-GS’s dataset, it is hard to fully confirm whether the module can effectively adapt to existing HDR-related data distributions, nor can it more comprehensively demonstrate the module’s advantages over tone-mapping components in static HDR NVS methods. This gap makes the module’s comparative advantage in the broader HDR tone-mapping context less fully validated .

### Questions
same as weakness.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper proposes the High Dynamic Range Dynamic Novel View Synthesis (HDR DNVS) task, aiming to reconstruct spatiotemporally consistent HDR radiance fields and dynamic geometry from sparse, time-varying LDR inputs. It introduces the HDR-4DGS framework based on Gaussian Splatting, featuring a dynamic tone-mapping module (DTM) to bridge HDR-LDR domains while maintaining temporal radiance coherence. Two datasets (HDR-4D-Syn with 8 synthetic scenes and HDR-4D-Real with 4 real indoor scenes) are constructed.

### Strengths
1.	The DTM, with a radiance bank and dynamic radiance context learner (DRCL), maintains spatiotemporal consistency, validated by ablation studies .
2.	Constructs dedicated datasets (HDR-4D-Syn/Real) with HDR ground truth and time-varying 3D geometry, enabling standardized evaluation .

### Weaknesses
1.	Given that most existing Dynamic Novel View Synthesis (DNVS) methods operate on low dynamic range (LDR) imagery and thus fail under high-contrast or temporally varying lighting, do you consider dynamic HDR modeling essential for achieving photorealistic results in real-world dynamic scenes? In other words, is extending DNVS to HDR space a fundamental requirement to overcome the perceptual and photometric limitations of LDR-based systems, rather than just an optional enhancement?
2.	No comparison with "dynamic scene reconstruction + independent HDR module" combinations, lacking proof of irreplaceability.
3.	When evaluating HDR-4DGS on the HDR-4D-Real dataset, the input LDR images are captured with six synchronized iPhone 14 Pro devices under three exposure times. However, real-world dynamic HDR scenarios may involve more diverse camera models (with different CRFs) or unsynchronized multi-view capture. Why is only a single type of synchronized device used for data collection, and has the method been tested for robustness to camera model differences or capture asynchrony?
4.	In the ablation study on temporal context length (k=20 as optimal), the paper only tests k=5,10,20,30. However, dynamic scenes with varying motion speeds (e.g., fast-moving objects vs. slow illumination changes) may require different k values. Did the authors design experiments to verify whether the optimal k=20 is adaptive to scenes with different dynamic intensities, or is it a fixed hyperparameter that needs manual adjustment for specific scenes?
5.	Is the proposed method able to address issues like extreme lighting performance?

### Questions
Please refer to the weakness part.

### Soundness
2

### Presentation
2

### Contribution
2

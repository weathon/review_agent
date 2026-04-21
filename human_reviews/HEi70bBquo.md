# Signal Dynamics in Diffusion Models: Enhancing Text-to-Image Alignment through Step Selection

- Avg Score: 4.33
- Decision: Reject
- Scores: 5, 3, 5

## Abstract
Visual generative AI models often encounter challenges related to text-image alignment and reasoning limitations. This paper presents a novel method for selectively enhancing the signal at critical diffusion steps, optimizing image generation based on input semantics. Our approach addresses the shortcomings of early-stage signal modifications, demonstrating that adjustments made at later stages yield superior results. We conduct extensive experiments to validate the effectiveness of our method in producing semantically aligned images, achieving state-of-the-art performance. Our results highlight the importance of a judicious choice of sampling stage to improve diffusion performance and overall image alignment.

## Human Reviews

## Human Reviewer 1

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper proposes an approach to improving text-image alignment that optimizes only one specific step during the generation process. Based on the analysis of experimental results, the authors show that later optimization brings better results. Extensive experiments are conducted to demonstrate the effectiveness of the methods.

### Strengths
1. The paper is well written.
2. The method is simple, clear, and easy to follow.
3. The experiments are extensive and reproducible.

### Weaknesses
1. The qualitative results are insufficient to support the stated improvement: 
e.g., in Fig. 1 case "a photo of a car and a blue cat", the attribute blue is still leaked to entity car; 
        in Fig. 1 case "a photo of a giraffe and a bear", the bear is still not well generated; 
        in Fig. 1 case "a photo of a giraffe and a banana", the number of bananas is wrong;
        in Fig. 6 case "a photo of a giraffe next to a car and a carrot", the carrots and the giraffe are mixed;
        in Fig. 6 case "a photo of a refrigerator next to a horse and a car", the location of each object is not reasonable compared with other methods.
2. The proposed method is incremental and lacks novelty, compared with Attend-and-Excite[1] and A-star[2].
3. The quantitative improvement is marginal in most widely used metrics. The method outperforms others only in terms of TIAM[3], which is not a well-recognized evaluation metric yet.
4. The experiments are conducted on SD 1.4, which falls behind current advanced models like SDXL, SD 3, and Pixart. Perhaps, the phenomenon in SD 1.4 and SDXL are different. The outdated base model makes this research less practical.

[1]Hila Chefer, Yuval Alaluf, Yael Vinker, Lior Wolf, and Daniel Cohen-Or. Attend-and-excite: Attention-based semantic guidance for text-to-image diffusion models. ACM Trans. Graph., 42(4), jul 2023. ISSN 0730-0301. doi: 10.1145/3592116. URL https://doi.org/10.1145/3592116.
[2] Aishwarya Agarwal, Srikrishna Karanam, K J Joseph, Apoorv Saxena, Koustava Goswami, and Balaji Vasan Srinivasan. A-star: Test-time attention segregation and retention for text-to-image synthesis. In Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV), pp. 2283–2293, October 2023.
[3]Paul Grimal, Hervé Le Borgne, Olivier Ferret, and Julien Tourille. Tiam - a metric for evaluating alignment in text-to-image generation. In Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pp. 2890–2899, January 2024.

### Questions
1. See Weakness 4. It will be better if some experimental results on more recent models are provided.
2. I would like to see the comparison of each method in terms of computational efficiency.
3. The paper is not well organized enough and the authors might consider modifying the figures and tables to better present their work.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
This paper introduce Signal Dynamics in Diffusion Models, which explore the key stage of signal modifications that could help to yield superior results.

### Strengths
1. The method selectively enhance the signal at a key diffusion step, optimizing image generation.
2. Explore and find the better stage to apply signal modification that leads to better results.

### Weaknesses
1. The paper lacks novelty. The main concept of the paper is to discover the best step to perform IterRef. It is not new and there are similar discoveries in FreeDoM[1].

2. The paper lacks the comparison between different methods, there are many other methods that could also achive higher text-alignment like RPG[2], SLD[3], the results in the paper are not competitive enough to support the claim of sota.

3. The current experimental analysis also appears insufficient. More evaluation metrics like FID,IS,T2I-CompBench[4] should be used to provide a more comprehensive results.

[1] Yu, Jiwen, et al. "Freedom: Training-free energy-guided conditional diffusion model." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2023.

[2] Yang, Ling, et al. "Mastering text-to-image diffusion: Recaptioning, planning, and generating with multimodal llms."

[3] Wu, Tsung-Han, et al. "Self-correcting llm-controlled diffusion models."

[4] Huang, Kaiyi, et al. "T2i-compbench: A comprehensive benchmark for open-world compositional text-to-image generation." Advances in Neural Information Processing Systems 36 (2023): 78723-78747.

### Questions
1. Can this method applied to different architectures like current sota models SD3 or FLUX ?

2. More evaluation results are needed between similar methods.

3. What is the fundemental difference between your findings and the results showed in Fig.3 of FreeDoM ?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
The paper titled "Signal Dynamics in Diffusion Models: Enhancing Text-to-Image Alignment through Step Selection" discusses a method to improve text-to-image alignment in generative AI models by enhancing signals at critical diffusion steps based on input semantics.  The study also highlights the importance of selecting the right diffusion steps for signal enhancement.

### Strengths
1. The assessment is comprehensive, indicating a thorough evaluation.
2. The performance of the proposed method surpasses that of previous work, as demonstrated by the comparative results presented in the paper's table.

### Weaknesses
1. The source code is inaccessible via the anonymous link provided.
2. The paper introduces numerous concepts without adequate explanation, which complicates comprehension:
* The term "GSN guidance" is not clearly defined. Is it a concept coined by the authors? If "GSN guidance" refers to the optimization of latents, why isn't the proposal method considered a form of "GSN guidance"? What is the distinction between "Ours" and "Ours+"?
* The phrase "lacks inherent semantic meaning" is used but not further elaborated upon.
* It is unclear what the "Similarity Score" is intended to measure.
3. The paper's novelty is questionable. The optimal iteration appears to be simply a result of the author's experimentation with various step iterations. But it has been widely accepted in research that detailed results are typically generated after the 8th step.

### Questions
Refer to the Weaknesses.

### Soundness
2

### Presentation
2

### Contribution
2

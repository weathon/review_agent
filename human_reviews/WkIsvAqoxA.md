# Dolfin: Diffusion Layout Transformers without Autoencoder

- Avg Score: 3.67
- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3

## Abstract
In this paper, we introduce a novel generative model, Diffusion Layout Transformers without Autoencoder (Dolfin), which significantly improves the modeling capability with reduced complexity compared to existing methods. Dolfin employs a Transformer-based diffusion process to model layout generation. In addition to an efficient bi-directional (non-causal joint) sequence representation, we further propose an autoregressive diffusion model (Dolfin-AR) that is especially adept at capturing rich semantic correlations, such as alignment, size, overlap, and neighborhood, between layout items/elements. When evaluated against standard generative layout benchmarks, Dolfin notably improves performance across various metrics, enhancing transparency and interoperability in the process. Moreover, Dolfin's applications extend beyond layout generation, making it suitable for modeling generative geometric structures, such as line segments. Our experiments present both qualitative and quantitative results to demonstrate the advantages of Dolfin.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
This paper introduces a diffusion-based layout generation model utilizing transformers. It removes the autoencoder layer typically used in a diffusion model for layout/image generation and directly operates on the layout input space. The proposed two model variants (Dolfin and Dolfin-AR) empirically improves performance across various metrics.

### Strengths
1. This paper is clearly written and easy to follow. 
2. The proposed models notably improve quantitative results against generative layout benchmarks.

### Weaknesses
1. The main difference with previous models is by operating directly on the input space of layouts (the coordinates and corresponding class labels) instead of processing the layouts with VAE/dedicated modules. However the reasons for the brought-in performance gains are not sufficiently justified.  
2. "enhancing transparency and interoperability" is overclaimed since it is a property of the standard diffusion process itself. 
3. From the paper presentation it is not clear what are the modifications to the original DiT transformer other than omitting a category input.

### Questions
1. Why is operating on the original layout space better, especially when processing such data with dedicated neural modules is quite standard ? e.g. other than mentioned related works also standard in other generative models such as [1]. Could it be that the training data is insufficient? 
2. Please check the metric arrow directions in Table 3, 4, 5.
3. In Fig.6, the generated samples exhibit some obvious unnaturalness (e.g. blue frames, bottom left, the window lines). Similar patterns exist in Fig.13. Is it because of insufficient training ? Could you compare it with PLay? 

[1] GLIGEN: Open-Set Grounded Text-to-Image Generation

### Soundness
2 fair

### Presentation
2 fair

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
Diffusion Layout Transformers without Autoencoder (Dolfin) is proposed, with an efficient bi-directional (non-causal joint) sequence representation. An autoregressive diffusion model (Dolfin-AR) is also proposed to capture rich semantic correlations for the neighboring objects. The method is validated on 2D layout generation and line segment generation tasks.

### Strengths
- not requiring the autoencoder layer in the diffusion model
- autoregressive diffusion model to capture the rich semantic correlation between objects/items
- experiment on generating geometric structures beyond layout, such as line segments

### Weaknesses
- not using auto encoder is not a new idea, Imagen model is processing directly on pixels
- there is no intuition on why auot-regressive design leads to better semantic correlation, although this is observed from experiments
- not many baselines comparison for the line segment generation

### Questions
- explain the intuition of the advantage of auot-regressive design
- compare with image diffusion results for line generation, line representation can be obtained followed by a line detector
- each object in a layout is represented by a 4 × 4 tensor, why we need 4 entires for the entire layout width/height? Once it's normalized, is that always -1/1?
- in Algorithm 1 and 2, is it better to use a different index than "t" in the for loop? The for loop index has different meaning than the diffusion step t.

### Soundness
3 good

### Presentation
3 good

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
4: You are confident in your assessment, but not absolutely certain. It is unlikely, but not impossible, that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work.

### Summary
The paper introduces "Dolfin," a generative model that uses a transformer-based diffusion process for layout generation. The proposed method directly applies on the input space of the geometric objects. The method benefits from bi-directional representation and consists of two versions, the non-auto regressive version that process all tokens simultaneously and the auto-regressive version that predicts each token sequentially. The authors provided experiments on RICO and PublayNet datasets for layout generation task as well as additional experiments on Line Segments Generation.

### Strengths
The paper is detailed and easy to follow.

Additional experiments on line segment generation can be useful to consider along with the other tasks.

### Weaknesses
The paper offers potential value to the community. However, concerns regarding its novelty and the robustness of its experimental evaluations need to be addressed for it to be ready for publication.

Novelty: The core proposition of the paper, which involves the utilization of the input coordinate space for layout design generation through continuous diffusion models, is not entirely novel. Similar approaches have been discussed in prior works such as [1, 2].

Experiments and Comparison: The experiments presented currently lack comprehensiveness. The results in Table 1 do not facilitate a fair comparison between the proposed method and existing methods. Although Tables 2 and 3 provide more data points, they restrict their focus to MaxIoU and Alignment scores. Furthermore, the results suggest that the proposed method underperforms compared to the baselines. This underscores the need for a more in-depth analysis and comparison.


[1] LEGO-Net: Learning Regular Rearrangements of Objects in Rooms, CVPR 2023
[2] HouseDiffusion: Vector Floorplan Generation via a Diffusion Model with Discrete and Continuous Denoising, CVPR 2023

### Questions
Considering the large batch sizes which are used in the experiments (10k and 6k) compared to the conventional batch sizes up to 2048, adding a table on the effect of the batch size on the final result can be very insightful.

I couldn't find any direct comparison on pros and cons of the Dolfin and Dolfin-AR. It is better if you also add both versions to other tables as well.

### Soundness
2 fair

### Presentation
2 fair

### Contribution
1 poor

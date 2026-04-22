# Scaling Group Inference for Diverse and High-Quality Generation

- Avg Score: 5.00
- Decision: Accept (Poster)
- Scores: 6, 6, 4, 4

## Abstract
Generative models typically sample outputs independently, and recent inference-time guidance and scaling algorithms focus on improving the quality of individual samples. However, in real-world applications, users are often presented with a set of multiple images (e.g., 4-8) for each prompt, where independent sampling tends to lead to redundant results, limiting user choices and hindering idea exploration. In this work, we introduce a scalable group inference method that improves both the diversity and quality of a group of samples. We formulate group inference as a quadratic integer assignment problem: candidate outputs are modeled as graph nodes, and a subset is selected to optimize sample quality (unary term) while maximizing group diversity (binary term). To substantially improve runtime efficiency, we use intermediate predictions of the final sample at each step to progressively prune the candidate set, allowing our method to scale up efficiently to large input candidate sets. Extensive experiments show that our method significantly improves group diversity and quality compared to independent sampling baselines and recent inference algorithms. Our framework generalizes across a wide range of tasks, including text-to-image, image-to-image, and image prompting, enabling generative models to treat multiple outputs as cohesive groups rather than independent samples.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
2

### Summary
This paper studies the task of batch inference with both quality and group diversity constraints. The authors formulate this problem as a quadratic integer program. To scale the method to large batch sizes, the authors propose a method that progressively prunes the candidate set of generations based on intermediate predictions of their final quality and final group diversity constraints. The authors conduct thorough experiments showing that their method both qualitatively and quantitatively outperforms existing batch inference methods.

### Strengths
- The paper is really well-written and easy to follow. As a non-expert in this area, I commend the authors for how smooth the paper reads!
- The proposed method is simple, natural, and seemingly efficient (I view this as a strength!). More overall, its general enough to capture quality and diversity constraints beyond the ones that the authors focus on in the main text. 
- The experimental results seem very promising -- the proposed method clearly dominates existing batch inference methods in terms of both quality and efficiency, based on Figures 4 and Figures 6. However, I am not an expert in this field, and it is very likely that I am not aware of other baselines beyond the ones the authors have considered. 
-  I also liked that the authors conducted a user study. One suggestion for Table 1 is that it seems like the baseline pref. column is redundant (since its just one minus the "ours pref" column). For clarity, I would remove that.

### Weaknesses
I don't have any particular concerns. However, I do feel that this paper would benefit from providing more motivation for why one should care about the batch inference problem studied. Right now, the only motivation is provided in Lines 37 - 42, however, I felt a bit unsatisfied. It would be great if the authors can further expound about why I should care about this problem.

### Questions
See Weaknesses.

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
They propose scalable group inference method which improves the diversity and quality of generated outputs. They view this problem as quadratic integer programming problem (QIP), and they leverage intermediate predictions from each denoising steps for efficient filtering. To provide validity, they do experiment on accuracy of intermediate predictions, and provide computational complexity of its algorithm. Its results show better trade-off in diversity and quality than naive baselines, such as Low-CFG, and related works, such as Interval Guidance and Particle Guidance. Its algorithm can be extended to unique objectives, such as color diversity.

### Strengths
S1. The paper is well-structured and clearly presented, with a solid experimental evaluation that thoroughly demonstrates all experiments substantiating the proposed method. The paper is well-organized and easy to follow.

S2. The problem addressed in the paper is highly relevant to real-world scenarios, and the proposed method effectively resolves it without requiring any additional training.

S3. The supplementary material is highly detailed, including numerous applications and qualitative results. It also provides reproducible implementation details and actual code.

### Weaknesses
W1. Limited novelty of proposed method. Proposed method is too straightforward. The idea of leveraging intermediate prediction accuracy is already well established, and the algorithm simply performs pruning based on group properties.

W2. Error in line 455: the original inference-time scaling paper (orange) [1] also incorporates intermediate prediction in its algorithm (see page 8). 

W3. Although the proposed method is effective for scaled group inference scenarios, it relies on a pruning (filtering)-based approach rather than directly guiding the generation process (e.g., explicit guidance). Therefore, it may be less suitable than other baselines when performing additional small-budget sampling with given images (pre-sampled or user-provided), particularly for rare cases within the generation pool.

[1] Ma et al., Inference-Time Scaling for Diffusion Models beyond Scaling Denoising Steps, Arxiv version

### Questions
Q1. In Figure 7, the progressive pruning in FLUX-dev does not seem to reduce the runtime as effectively as expected. Is there a specific reason for this difference compared to Schnell?

Q2. Better diversity through negative guidance[2] could also serve as a baseline. While your gradient-free method seems more effective in terms of quality, I am curious about how it compares in terms of runtime.

[2] Singh et al, Negative Token Merging: Image-based Adversarial Feature Guidance, Arxiv

### Soundness
3

### Presentation
4

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
This paper proposes a group inference approach for diffusion-based generative models to improve the diversity of generated samples. To this end, the authors formulate the generation of multiple outputs as a Quadratic Integer Programming (QIP) problem, where unary terms represent sample quality and binary terms represent diversity among samples. To address computational inefficiency, they introduce a progressive pruning mechanism that leverages intermediate denoising predictions to discard low-quality candidates early. Experiments across various tasks show improvements in both diversity and quality compared to several baselines.

### Strengths
- The paper addresses an important problem of improving diversity of modern diffusion models, which excels at producing high-quality images yet suffers from lack of variations.
- It is interesting to cast diversity-enhancing inference as a QIP problem.

### Weaknesses
- The related work section omits several important and relevant diversity-promoting diffusion methods, such as CADS [1], Shielded Diffusion [2], and DiversityFlow [3].  
- Experimental comparison with existing baselines is limited.  
  1. The paper only compares with Particle Guidance (PG) and Interval Guidance (IG). More diversity-oriented baselines such as CADS [1], Shielded Diffusion [2], and DiversityFlow [3] should be incorporated to clarify the empirical benefits of the proposed approach.  
  2. There is no runtime comparison against baselines. Since the proposed method involves seemingly-expensive additional steps (during QIP solving and pruning), it is important to quantify the runtime overhead relative to existing methods. This is a significant point, since some diversity approaches (like CADS, IG, and Shielded Diffusion) are known to introduce marginal computations for improving diversity.
- The evaluation metrics are incomplete.  
  1. The diversity metric should include established measures like Vendi Score [4] or MSS [1], which are standard in diversity-oriented generation research [1-3].  
  2. The quality evaluation relies only on ImageReward and CLIPScore, which are often highly correlated. Including other metrics such as PickScore or HPSv2 would provide a more comprehensive assessment. 
  3. A standard comprehensive metric like FID is missing.

---
**References**

[1] CADS: Unleashing the Diversity of Diffusion Models through Condition-Annealed Sampling, ICLR 2024.

[2] Shielded Diffusion: Generating Novel and Diverse Images using Sparse Repellency, ICML 2025.

[3] DiverseFlow: Sample-Efficient Diverse Mode Coverage in Flows, CVPR 2025.

[4] The Vendi Score: A Diversity Evaluation Metric for Machine Learning, Arxiv 2022.

### Questions
See weaknesses

### Soundness
2

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
5

### Summary
The paper proposes a method for generating diverse and high-quality image groups. The approach begins with a large pool of candidate initial noises and gradually prunes this pool by solving a QIP. The final subset of images is selected to jointly optimize quality and diversity. The evaluation is primarily based on human preference studies, which indicate improvements in both diversity and perceived quality.

### Strengths
- The paper addresses the practically relevant setting of generating a group of images rather than a single sample, which is common in many real-world applications.
- The proposed selection strategy leads to substantial gains over baselines, according to human evaluations. The visual examples clearly illustrate the improvements of the method.
- The paper is clearly written and easy to follow.

### Weaknesses
- The method is significantly slower than the baseline that directly generates four images. According to the paper, the proposed pipeline takes approximately 2.5 times longer. Given the same time, a straightforward baseline can generate approximately ten images, while the proposed method generates only four.
- The method does not scale efficiently with respect to the number of candidates $M$ and the pruning ratio $p$. QIP solving grows quickly with $M$, and the required decoding and scoring introduce further overhead. According to Figure 9, the growth rate is considerably steeper than that of direct generation. This could sufficiently limit the method’s applicability.

### Questions
- According to the described procedure, most candidates are rejected at the timesteps with the lowest correlation with the final image. Could the authors provide an ablation where no pruning is done until the last stage, and the QIP is solved only once at the final step with $M = 64$ candidates and $K = 4$ outputs? How does this compare to the default setting with $p = 0.5$? 
- Could the authors provide a comparison under the same time budget, where the naive baseline produces 10 images and the proposed method produces 4? Could the authors report results for a setting with only 10 candidates, $p = 1.0$, and a single QIP solve at the final step for 4 outputs?
- Could the authors report results using alternative unary score terms (i. e. HPSv2[1], ImageReward[2]) instead of CLIP similarity? These reward models are explicitly trained to optimize human visual preference. How sensitive is the final selection to the choice of unary score metric?
- Could the authors clarify the exact definition of the combined score? In particular, is this score meant to be maximized or minimized? How are the different components combined to produce this score?

### Soundness
3

### Presentation
3

### Contribution
3

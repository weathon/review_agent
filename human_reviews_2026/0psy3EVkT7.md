# GeoMoLa: Geometry-Aware Motion Latents for Learning Robust Manipulation Policies

- Avg Score: 4.00
- Decision: Reject
- Scores: 4, 2, 6

## Abstract
Learning motion latents for robotic manipulation heavily relies on extracting motion patterns from visual sequences, yet effective action abstractions require understanding three-dimensional geometric transformations. Here, we introduce GeoMoLa (Geometry-Aware Motion Latents), which learns discrete motion latent codes by predicting how point clouds evolve during manipulation rather than reconstructing
visual observations. This four-dimensional objective – spatial geometry changing through time – forces latent representations to encode actual physical motion rather than appearance patterns. GeoMoLa achieves state-of-the-art performance using only single-view RGB-D input, while existing methods require multi-view reconstruction, succeeding across diverse manipulation benchmarks. Our ablations
reveal that geometric prediction is the key to driving performance, quantitatively validating that manipulation depends on spatial understanding. Furthermore, the learned codes exhibit effective motion abstraction: applying them to novel scenes
produces physically consistent transformations regardless of visual context. Our real-world experiments also confirm this robustness capability, achieving robust manipulation with minimal demonstrations in cluttered environments where geometric reasoning determines success. Thus, we demonstrate that effective motion latents for robot control can better emerge from understanding motion through its
three-dimensional effects rather than pixel-level patterns.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
The paper proposes a self-supervised way of learning discrete action codes through predicting how point clouds evolve during manipulation rather than reconstructing visual observations. The paper argues that the spatial geometry changing through time is important and useful to spatial understanding. Experiments on different benchmarks show superior performances. The paper claims they are the first framework that models robot manipulation as continuous four-dimensional process.

### Strengths
* The proposed method only uses single-view RGB-D inputs, while still achieving competitive performance. This is important in real-world deployment of robots.
* The intuition behind the proposed method makes sense. The learned results show good interpretability also hints the effectiveness of the method.

### Weaknesses
* The methodology section is quite hard to follow. There is few logical connection between each subsection. The writing mostly consists of plain descriptions of models, without many explanation, which makes the framework hard to understand.
* The logical connection of Figure 2 is unclear. What is the relation between (a) and (b)? Do they share any module?
* It is hard to see how the four-dimensional geometry changing is imposed in the training objective. It looks like from line 223 the constraint is imposed through predicting future latent point map features. And the latent features are obtained through a finetuned diffusion model described in Appendix D. The point map VAE is initialized by an RGB VAE, which does not make too much sense to me since 3d coordinates are different modalities than RGB. And Appendix D has some undefined variables such as z^t.

Overall the writing of the paper seems a big issue for me. It is quite confusing so that I find it hard to evaluate the correctness of the architecture. Although the paper has good intuition and seemingly good results, I would rate a borderline rejection.

### Questions
Which specific loss function imposes the constraint that "encode actual physical motion rather than appearance patterns"?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
5

### Summary
The paper proposes GeoMoLa. It learns discrete motion codes by predicting future 3D geometry (pointmaps) and RGB from current RGB‑D observations and language. 

A VQ‑VAE discretizes vision‑language features into codes; a pointmap/RGB latent diffusion model is trained to forecast future observations conditioned on those codes; and a 3D denoising transformer uses the codes to generate 6‑DoF action chunks. 

Experiments on RLBench, CALVIN, and six real‑robot tasks show consistent gains over 2D/3D baselines, with ablations indicating geometry prediction is the main driver of performance.

### Strengths
1. Method is well motivated: VQ‑VAE for discrete codes; pointmap diffusion for future geometry; 3D‑aware transformer with relative 3D attention and cross‑attention to latents for action generation.

2. Good experimentation: solid benchmarks and real‑world evaluation with low demo counts, plus clean ablation identifying geometry prediction as the key contributor.

### Weaknesses
1. The main contribution to me is tying discrete latents to future 3D geometry prediction and then using them to condition 3D‑aware action diffusion. Might better tone down “first 4D” phrasing and draw distinction vs dynamic Gaussian / NeRF‑style approaches.

2. Baselines are not strong enough. For example, in RLBench experiments, RVT2 is not included. It gets 100% on close jar and 80% on stack block.

2. Besides, the presentation is not very clear. Latent motion / latent action is not consistent and thus confusing to readers. For example,  fig2 has (a) Geometry-Aware Latent Action Learning, and (b) Latent-Conditioned Action Generation. If I get it right, the latent action is to motion latents in the title. But it could mean latent embedding learned for robot action space. Thereby it is unclear to the latent action learning actually until reading much more in depth.

### Questions
What is the motivation of deriving latent action from rib and language using minigpt? 
Would it be more natural to have depth / point map as input to latent action as well? considering they are assumed available in both training and inference

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
4

### Summary
The presented work introduces a novel approach to learn latent actions. Rather than learning the latent actions to predict future images, the authors propose learning the latent action to predict both future RGB and pointclouds. Thanks to the pointcloud prediction, the authors claim that the latent actions better capture the 3D geometric of the task. 

Thanks to a better geometric component in the learned latent actions; when exploited for policy learning, the learned policies lead to better policies (higher task success rates).

The performance of the model was evaluated both in simulation (CALVIN, RLBench) and real robot. The authors also present ablation studies on the impact of learning the latent actions with and without pointcloud prediction and with and without RGB prediction.

### Strengths
**Originality**
The presented work is original in learning latent action representations with the additional 3D geometric embeddings. While previous works [1] propose learning the latent action embeddings with only future RGB prediction, the presented work proposes learning the embedding with both RGB and pointcloud. 
The authors argue that the pointcloud might led to better capturing the geometry, a reasoning that is sound.

**Quality**
The work makes a good job in evaluating the performance of the model under multiple evaluations and present a useful ablation to visualize the real impact of adding 3D geometric prediction in the latent action pretraining.

**Clarity**
The work is easy to read and to follow.

[1] Ye, S., Jang, J., Jeon, B., Joo, S., Yang, J., Peng, B., ... & Seo, M. (2024). Latent action pretraining from videos. arXiv preprint arXiv:2410.11758.

### Weaknesses
- Weak improvements due to the 3D geometry. While the authors show in Tab 3., a performance improvement thanks to the 3D geometry, the improvements are not large (2% increase in RL Bench and max 5% in CALVIN). Also the variations on the performance increase among tasks, makes it wonder when does the 3D geometry helps and when does not. Authors could consider exploring some simple “demo tasks”, one where 3D geometry does not help and one where 3D geometry is essential and find out if the latent embeddings with the 3D geometry is useful.
- Another interesting analysis could be done in comparing the performance enhacement individually in tasks that require 3D translations informations and tasks that require rotation information. Is the pointcloud-based latent action embeddings equally useful for both?
- Figure 1 is not very informative. Being the first figure of the paper, authors could try to improve the first figure to give a better grasp of the main idea. While it is able to give the general idea of “3D better”, it lacks details and it is too general to be valuable. Authors could consider including information regarding the latent action embedding and how is different from previous latent action embeddings.
- 3D diffuser actor reported performance is lower than in their paper. While the original paper claim an average success rate of 81.3%, in your work the performance is 77%. Is there a reason for this mismatch?

### Questions
- 3D diffuser actor reported performance is lower than in their paper. While the original paper claim an average success rate of 81.3%, in your work the performance is 77%. Is there a reason for this mismatch?

- What could be the reason of observing not very large performance enhancement when training the latent actions with pointcloud prediction?

### Soundness
3

### Presentation
2

### Contribution
2

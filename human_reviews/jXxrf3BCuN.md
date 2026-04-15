# Inference from Real-World Sparse Measurements

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 6, 3

## Abstract
Real-world problems often involve complex and unstructured sets of measurements, which happens when sensors are sparsely placed in either space or time. Being able to model this irregular spatiotemporal data and extract meaningful forecasts is crucial. Deep learning architectures capable of processing sets of measurements with positions varying from set to set, and extracting readouts anywhere are methodologically difficult. Current state-of-the-art models are graph neural networks and require domain-specific knowledge for proper setup.

  We propose an attention-based model focused on robustness and practical applicability, with two key design contributions. First, we adopt a ViT-like transformer that takes both context points and read-out positions as inputs, eliminating the need for an encoder-decoder structure. Second, we use a unified method for encoding both context and read-out positions. This approach is intentionally straightforward and integrates well with other systems. Compared to existing approaches, our model is simpler, requires less specialized knowledge, and does not suffer from a problematic bottleneck effect, all of which contribute to superior performance.

  We conduct in-depth ablation studies that characterize this problematic bottleneck in the latent representations of alternative models that inhibit information utilization and impede training efficiency. We also perform experiments across various problem domains, including high-altitude wind nowcasting, two-day weather forecasting, fluid dynamics, and heat diffusion. Our attention-based model consistently outperforms state-of-the-art models in handling irregularly sampled data. Notably, our model reduces the root mean square error (RMSE) for wind nowcasting from 9.24 to 7.98 and for heat diffusion tasks from 0.126 to 0.084.

## Human Reviews

## Human Reviewer 1

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper proposes an attention-based method for handling irregularly sampled data. The proposed method uses ViT-like transformer architecture to combine context points and read-out positions. Compared with other baseline models, the proposed method shows better performance in data forecasting tasks based on irregularly sampled data.

### Strengths
- This paper tackles irregularly sampled datasets, which is an under-explored area in the scientific ML community. 

- The proposed method is simple and powerful. It is applicable to data from different domains.

### Weaknesses
- The presentation of this paper can be optimized. I think the description for baseline models can be shrunken, and the authors may expand the introduction of their proposed method in the Methodology section. 

- The Experiments part is kind of weak, in terms of evaluation metrics. This paper only considers RMSE. For ERA5 datasets, scientists usually conduct comprehensive evaluations from different perspectives [1-2]. For scientific data (heat diffusion, and fluid dynamics), relative L2 errors are commonly considered. Adding more evaluation metrics can strengthen this paper.

- The writing of this paper is not good enough. There are many typos and grammatical issues. I list several of the issues below. Please do a thorough check. 
    - On Page 1, “In contrast, various real-world data is often comes from …”, “is” is redundant. 
    - On Page 1, “An example which can benefits significantly…”, “benefits” should be “benefit”.  
    - On Page 2, “MSA keep one latent vector per encoded measurement and can access them directly for forecasting.”, “keep” should be “keeps”. 

---

**Refs:**

[1] Bi, Kaifeng, et al. "Pangu-weather: A 3d high-resolution model for fast and accurate global weather forecast." arXiv preprint arXiv:2211.02556 (2022).

[2] Pathak, Jaideep, et al. "Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators." arXiv preprint arXiv:2202.11214 (2022).

### Questions
What spatial subsampling methods do the authors use in this paper? It seems not mentioned in the paper.

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
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
In this paper, the authors propose a novel attention mechanism allowing to efficiently apply attention layers to data on non regular grids. The authors demonstrate the effectiveness of the proposed scheme on several applications ranging from PDEs to nowcasting.

### Strengths
- The paper is well written;
- The authors demonstrate the soundness of the proposed method on several problems;
- Experiments are convincing;
- In particular, results on the wind nowcasting problem are impressive.

### Weaknesses
- The link with the literature is unclear making the novelty of the approach difficult to assess. In particular, works related to motion forecasting (which shows strong similarities with the reference example of this paper) are not referenced [1, 2, 3]. In a context of explosion of attention-based architectures, a more detailed literature review is expected.
- The global presentation of the attention mechanism does not follow the same notations and presentation as is usually the case in the literature and could gained in being clarified.
- The authors alternate between the presentation of a general attention mechanism and a very specific application (wind nowcasting), making it difficult to keep in sight the goal of the paper.

**References** 

[1] Nayakanti, Nigamaa, et al. "Wayformer: Motion forecasting via simple & efficient attention networks." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.
[2] Yuan, Ye, et al. "Agentformer: Agent-aware transformers for socio-temporal multi-agent forecasting." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.
[3] Girgis, Roger, et al. "Latent variable sequential set transformers for joint multi-agent motion prediction." arXiv preprint arXiv:2104.00563 (2021).

### Questions
I thank the authors for their interesting work.

**Major comments**

1. My first concern is the clarity of the setup. In particular, I found section 3.1 confusing. The apparent will to keep very general notations, while being motivated by a very specific example that not all readers of ICLR papers are familiar with, is misleading. The authors write: "prediction of target values given [...] context and [...] target position. [...] (c_x, t_x) are positions [...] "(c_y, t_y) are measurements." The notations seem to suggest that "c_x" stands for "context at coordinate x" and t_x for "target at coordinate x", but then the network inputs are (c_x, c_y, t_x). An illustration would be very welcome at this stage. Why not move one of the figures in the supplementary at that level? This would clarify the presentation greatly. (Typically, the following sentence : "The context consists of past wind speed measurements, while the targets comprised subsequent wind speed measurements taken at potentially different positions." rather induced more misunderstanding on my side than any clarification. I had to rely on Figure 4 to understand the underlying problematic. Figure 3 is very clear, too.)
2. I find that the notations chosen in equations (1)-(9) is rather uncommon and make it difficult to understand the novelty of the approach. Are the $\varphi, \psi, \phi, \nu$ the usual $\operatorname{softmax}(QK^\top)V$ layers? Integrating (3) and (4) to (5) and (6) would clarify the approach. Similarly, eq. (10)-(13) could be moved to supplementary. 
3. Is the proposed method only removing a layer to TFS, as suggested after (9)?
4. It remains unclear to me how the proposed work relates to other works such as [1-5]. Could the authors elaborate on that?
5. "Current state-of-the-art models are graph neural networks and require domain-specific knowledge for proper setup." Graph neural networks are not the only ones employed on spherical data, see e.g. [6, 7]. Could the authors comment on that?
6. Similarly, for irregularly sampled data, Fourier Neural Operators are an option too [8] that have been applied to weather forecasting [9]. Could the authors comment on that?

**Minor comments**

1. I think the abstract could be synthetized in a single paragraph; the last sentence is not necessary.
2. Table 1 is very vague as it is not quantitative (how do the authros evaluate "simplicity"?) and does not adds much to the paper.
1. "as it does not create a bottleneck." --> I think it would be good to define there what the authors mean by bottleneck.
2. "ablations studies" --> "ablation studies"


**References**

[1] Nayakanti, Nigamaa, et al. "Wayformer: Motion forecasting via simple & efficient attention networks." 2023 IEEE International Conference on Robotics and Automation (ICRA). IEEE, 2023.

[2] Yuan, Ye, et al. "Agentformer: Agent-aware transformers for socio-temporal multi-agent forecasting." Proceedings of the IEEE/CVF International Conference on Computer Vision. 2021.

[3] Girgis, Roger, et al. "Latent variable sequential set transformers for joint multi-agent motion prediction." arXiv preprint arXiv:2104.00563 (2021).

[4] Alerskans, Emy, et al. "A transformer neural network for predicting near‐surface temperature." Meteorological Applications 29.5 (2022): e2098.

[5] Bilgin, Onur, et al. "TENT: Tensorized encoder transformer for temperature forecasting." arXiv preprint arXiv:2106.14742 (2021).

[6] Cohen, Taco S., et al. "Spherical cnns." arXiv preprint arXiv:1801.10130 (2018).

[7] Ocampo, Jeremy, Matthew A. Price, and Jason D. McEwen. "Scalable and equivariant spherical CNNs by discrete-continuous (DISCO) convolutions." arXiv preprint arXiv:2209.13603 (2022).

[8] Li, Zongyi, et al. "Fourier neural operator for parametric partial differential equations." arXiv preprint arXiv:2010.08895 (2020).

[9] Pathak, Jaideep, et al. "Fourcastnet: A global data-driven high-resolution weather model using adaptive fourier neural operators." arXiv preprint arXiv:2202.11214 (2022).

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

---

## Human Reviewer 3

### Rating
6: marginally above the acceptance threshold

### Rating Number
6

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
This paper tackles making predictions in irregularly sampled (in space and time) sequence modeling tasks.  The net result is a transformer-like architecture that can gracefully scale and adapt to the irregularities.  Some comparisons are then made between the architectures in a synthetic task to understand where the performance gains may be coming from.

### Strengths
I think this paper is a nice application paper.  Clearly flexible function approximators are needed for weather forecasting from sparse mobile sensors.  The empirical validation also appears sound, and some of the experiments pulling apart the architectures in Section 5 are a great complement.  The paper itself is also reasonably well written and reads very easily.  Figures are nicely prepared and support the textual exposition.  The supplementary materials and code (although I didn’t actually run it) are incredibly thorough.

### Weaknesses
I found this a really difficult paper to review.  The paper is enjoyable to read, fairly self-contained, and the experiments seem to support the hypothesis.  Presenting a simple method _should not_ present a barrier to publication.  This paper is also an absolute A+ in terms of clarity and supporting materials.  Kudos to the authors.  

My only real concrete criticism is that I think the technical contribution is minor.  Separating the input encoder is not exactly an awe-inspiring innovation – but it seems to work well on the experiments provided, so I have a hard time pushing back too much on it.  Some of the ablation experiments are nice, but they are not exactly surprising to me – bottleneck layers limit recall necessarily, attention provides clear computation pathways, etc.  But these are still interesting, correct and well-presented investigations. 

The opening of the paper discusses a real-world need, but this need is subsequently unaddressed, which makes me think it was just some “window dressing”.  If the authors could propose a concrete application on real-world data, where the predictions from other methods could be verified as being “inadequate” by an understood metric, then I would find that compelling.  

I suppose my main lingering concern is simply one of impact.  I don’t believe the experiments here will convince many practitioners to adopt this method, or that the insight required for this method is dramatically different to simple “a transformer”.  The main “deep” baselines have all also been proposed by the authors, and so it is difficult to know if the MSA method is better, or if the baselines have just been poorly tuned.  Comparing to published literature removes this uncertainty.  However, again, I concede that I do not know what experiments _would_ convince me of this, or to what papers I would look to benchmark against.  Maybe comparison to examples in [1-3] would be compelling?  

Maybe comparatively then, I think the strength of the technical contribution, range and depth of the experimental evaluation, and general impact of this paper are lower than other papers I have recommended be accepted.  While this does not mean I recommend the paper be rejected, I find it difficult to ardently campaign for its inclusion.  

I invite the authors to push back against observations, particularly highlighting the innovations over existing models in the field or the general applicability of the solution

[1]  Learning Temporal Evolution of Spatial Dependence, Lan, PMLR, 2022.

[2]  NEURAL SPATIO-TEMPORAL POINT PROCESSES, Chen+, ICLR, 2021.

[3] A spatio-temporal statistical model to analyze COVID-19 spread in the USA, Rawat+, J Appl Stat, 2021.

### Questions
*Q.1.*:  Can the authors clarify what the “transformer-decoder” and “transformer-encoder” architectures are.

*Q.2.*:  Why do the “transformer-encoder” in (5) and (7) have different input and return types?

### Soundness
3 good

### Presentation
3 good

### Contribution
1 poor

---

## Human Reviewer 4

### Rating
3: reject, not good enough

### Rating Number
3

### Confidence
3: You are fairly confident in your assessment. It is possible that you did not understand some parts of the submission or that you are unfamiliar with some pieces of related work. Math/other details were not carefully checked.

### Summary
A modularized transformer is proposed, where context and target information is encoded in separate modules to process spatially irregularly distributed data. In a comparison against a graph neural network and conditional neural processes, the proposed method shows improved performance on a wide variety of datasets.

### Strengths
_Originality:_ Neither the task nor the method seem to be particularly new. The modularization of the transformer into different compontents is appealing, though.

_Quality:_ Given the conducted experiments, it is not all clear whether the proposed model holds what the authors promise and whether it is state-of-the-art. My biggest concern is the limited amount of related work cited in the manuscript and I do not see the field of related work well explored. There are numerous works that operate on irregular meshes, e.g., [[1]](http://arxiv.org/abs/2302.10803), [[2]](http://arxiv.org/abs/2207.05209), [[3]](http://arxiv.org/abs/2210.05495), [[4]](http://arxiv.org/abs/2210.00612), [[5]](http://arxiv.org/abs/2103.01342), which I would require to see in a comparison study.

_Clarity:_ I had great difficulties to understand large parts of the manuscript. In particular, the data description was unclear to me and the purpose of some experiments remained vague to me (see questions below)

_Significance:_ The significance of this method and its contribution to the field are unclear to me. The results on the wind forecasting, in particular, are promising but need more contextualization with other models accomplishing similar tasks.

_Further comments_:
- The manuscript comes with a very detailed appendix and supplementary material, which is a great resource for exploration! Adding the Navier-Stokes figure that shows context and target points to the main document would be helpful, for example.
- The number of benchmark datasets is great; ideally, this should be complemented with more competitive models form the litareture.

### Weaknesses
1. Several unclear details. What is the spatial and temporal distance of measurements in the different datasets? what do you mean with bottleneck? Do you refer to information compression when projecting to latent space (I could not find where it is defined)?
2. The naming of context and target was confusing to me. In Section 3.1 you specify: `Data is in the form of pairs of vectors $(c_x,c_y)$ and $(t_x,t_y)$ where $c_x$ and $t_x$ are the position and $c_y$ and $t_y$ are the measurements (or values).` It is unclear to me whether $x$ and $y$ are referred to inputs and targets or if $c$ and $t$ denote inputs (context) and targets. Maybe use $p_x, p_y$ for positions of input and target, respectively, and $v_x, v_y$ for the vectors holding the values for inputs and targets, respectively.
3. I am not sure whether the traditional transformer with positional encoding has been benchmarked here. Neither GEN nor CNP seem to resemble the traditional transformer. The vanilla transformer would constitute an indispensible baseline.
4. The context information retrieval experiment was difficult to grasp and I could not quite understand why other models (with a bottleneck) fail. Isn't the problem just a matter of spatial sampling? That is, when having sufficient point measurements, wouldn't all model perform well? Alternatively, wouln't a larger latent vector solve the issue as well?
5. I do not quite understand your statement in Section 5.1 `The higher $f$ is, the more difficult the function becomes, as local information becomes less informative about the output.` To my understanding, larger values of $f$ increase the frequency, making local information more important than far distant information.
6. Unclear what runtime and/or memory consumption the introduced method induces. According to Figure 5, MSA seems to employ a 8x8 grid, which might be expensive to handle computationally. Some information about computation, memory, and accuracy trade-offs would be helpful to justify the method. How big is the context size used in your experiments and when does the model break down (du to scaling, as mentioned in the last paragraph of the conclusion)?
7. Alhtough I do like the analysis, Figure 6 seems trivial to me. Doesn't the number of required gradient steps relate directly to the receptive field of the model and, hence, to the number of message passing steps required to propagate information through the grid?

### Questions
1. Given that MSA foregoes with any position information, how can it be able to predict wind direction and speed from surrounding sources? Wouldn't it be essential to know which winds are observed where in space?
2. What does the color code in Figure 4 resemble, is it an error?
3. Similarly, what does the x-axis in Figure 5 show, are these neurons in the output layer?
4. How do the benchmarked methods perform with different numbers of context points?

### Soundness
2 fair

### Presentation
2 fair

### Contribution
2 fair

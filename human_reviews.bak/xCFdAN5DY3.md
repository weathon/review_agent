# A Foundation Model for Weather and Climate

- Decision: Reject
- Scores: 6, 3, 5, 5

## Abstract
Triggered by the realization that AI emulators can rival the performance of traditional numerical weather prediction models running on HPC systems, there is now an increasing number of large AI models that address use cases such as forecasting, downscaling, or nowcasting. While the parallel developments in the AI literature focus on foundation models -- models that can be effectively tuned to address multiple, different use cases -- the developments on the weather and climate side largely focus on single-use cases with particular emphasis on mid-range forecasting. We close this gap by introducing Prithvi WxC, a 2.3 billion parameter foundation model developed using 160 variables from the Modern-Era Retrospective Analysis for Research and Applications, Version 2 (MERRA-2). Prithvi WxC employs an encoder-decoder-based architecture, incorporating concepts from various recent transformer models to effectively capture both regional and global dependencies in the input data. The model has been designed to accommodate large token counts to model weather phenomena in different topologies at fine resolutions. Furthermore, it is trained with a mixed objective that combines the paradigms of masked reconstruction with forecasting. We test the model on a set of challenging downstream tasks namely: Autoregressive rollout forecasting, downscaling, gravity wave flux parameterization, and extreme events estimation.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper titled "Basic Models for Weather and Climate" introduces Prithvi WxC, a 2.3 billion parameter basic model for various weather and climate tasks. These include downscaling, autoregressive prediction, and extreme event estimation. The model is trained on 160 variables from the MERRA-2 dataset and combines mask reconstruction with prediction tasks to learn from various atmospheric data. Its encoder decoder architecture and ability to work with different spatial topologies make it suitable for global and regional weather modeling.

### Strengths
Originality: The introduction of basic models for weather and climate applications is novel and important. Unlike specific task models such as FourCastNet and GraphCast (Lam et al., 2022), Prithvi WxC addresses a wide range of tasks and effectively narrows the gap between artificial intelligence models for specific weather tasks and general artificial intelligence base models.
Quality: The mixed pre training objective of this model combines masking and prediction, which is robust, especially because it uses climate bias rather than just future state prediction, which enhances the adaptability of the model. The model also showed impressive results in zero sample evaluation of reconstruction and autoregressive prediction, outperforming the baseline in a short delivery cycle.
Meaning: The ability to generalize to multiple downstream tasks, such as downscaling and gravity wave parameterization, suggests that this model has the potential to have a significant impact on weather and climate modeling.

### Weaknesses
Generalization to other datasets: Although the model performs well on MERRA-2 data, its generalization ability to other datasets such as ERA5 or CMIP has not been fully explored. Validation on different datasets will better demonstrate its robustness.
Long term prediction: The accuracy of Prithvi WxC decreases with the extension of the prediction window, especially beyond 66 hours, and its performance is poor compared to models such as Pangu. A deeper investigation into how to maintain performance within an extended time frame can improve its utility in medium - and long-term forecasting.

### Questions
Architectural flexibility: Can you provide a detailed explanation of the adaptability of the architecture when applied to non rectangular grid systems, such as those used for ocean simulation or polar regions?
Generalization strategy: What steps are taken to ensure the performance of the model when training or fine-tuning on datasets outside of MERRA-2, such as ERA5 or even higher resolution datasets?
Hurricane forecast: Can you provide more details on how the hurricane trajectory prediction of this model compares to specific task models such as FourCastNet, especially in areas with sparse data coverage?

### Soundness
3

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
The paper introduces a new foundation model for weather and climate, called Prithvi WxC, with 2.3 billion parameters, and trained on relatively unconventional yet interesting reanalysis products of MERRA-2. The authors use a relatively novel pre-training strategy in this field, where in addition to forecasting-only pre-training, they also combine masking for reconstruction in the hope of better self-supervision, among several upsides (e.g., natural extension to data assimilation with sparsely-gridded observation). The FM is then evaluated on several downstream tasks, including forecasting, downscaling, and parameterization.

### Strengths
Operationally, the large-scale pre-training of 2.3B FM is impressive in the emerging field of AI4Science. The use of large-scale dataset is also noteworthy. The writing is clear and the downstream tasks are clearly defined, and touch upon some of the hardest challenges facing the field. The use of beyond-atmospheric variables (coupled ocean + land) is also welcomed to build a full Earth system FM.

### Weaknesses
There are several major weaknesses in the paper, including non-existent/weak baselines and misleading claims, summarized below:

Major weaknesses
1. The term foundation model for this work is misleading, since in contrast to other similar works e.g., ClimaX, Aurora, the model is only trained on one data source which is MERRA-2. The problem: reanalysis products are by construction applicable for short-medium range applications as they are used either to evaluate NWPs or used as ICs for the next forecasting window. The climate dataset being missing, makes the claim of Prithvi being also a climate FM is an overreach at best. 

2. Related to the first point, the paper does not evaluate on any climate-related tests despite claiming it to be a climate FM too: (a) is the model stable over 50-100 years climate rollout? (b) what is the climate drift/bias compared to SOTA climate emulator? The paper applies downscaling to a climate dataset CORDEX, but this is less of a climate question and just a general downscaling problem since the former is more concerned about getting the long-term statistics, rather than a high-resolution state realization, correct (which is near impossible to nonlinear chaotic systems such as the Earth system). 

3. The downscaling benchmarking is also unacceptable as the baseline is too weak (e.g., bilinear, nearest neighbor). The author should either remove this part or add stronger baselines where the SOTA is at least a deep learning based method. Also, there is no benchmarking on the parameterization downstream task. At least use existing DL-based models from recent works. 

4. The forecasting performance appears to be unconvincing at best: with 2.3B parameter, it performs worse than e.g., <50M parameter GraphCast (~40x smaller) even at short lead time of 66 hours (<3 days). Even when the authors mention this result is "zero-shot" (which I find it unconvincing since there is rollout fine-tuning still) and the target is different (MERRA-2 vs ERA5), the obvious larger error growth (Figure 4) is alarming as it may not be useful for long-range climate forecasting. Also, why not benchmark against MERRA-FourCastNet in the forecasting task since this provides for a fairer comparison as both are trained with MERRA (as in the case for hurricane prediction). Finally, figure 4c (single line: forecasting cloud) is a case-in-point: the lack of sufficient apple-to-apple baselines where training/eval is done on MERRA-2. 

Overall, I find the results unconvincing given the lack of data sources, proper baselines, and inferior performance gain despite Prithvi being orders-of-magnitude larger than any SOTA model. As a side note: I believe the task of downscaling and parameterization is similar in that both attempts to resolve small-scale physics in an otherwise coarse-resolution model. I suggest the authors combine or use different downstream tasks e.g., climate projection.

### Questions
In addition to the weaknesses above, can the authors clarify the following:

1. How is the masking pre-training strategies better than just using variable lead-time that includes delta_t = 0? 
2. How is the local-global attention better than existing pre-training strategies of e.g., patch-variable-level tokenization employed in e.g., ClimaX? 
3. With masking as an additional pre-training strategy, what is the cost-performance tradeoff? 
4. Has the authors measure Prithvi's parameterization stability in an online-coupled setting?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
The paper presents Prithvi WxC, a new foundation model designed to support a wide range of weather and climate applications. Built on a large, transformer-based architecture, Prithvi WxC is trained on the extensive MERRA-2 dataset, which covers 160 variables capturing atmospheric data. The model is unique in its ability to address multiple tasks—including forecasting, downscaling, and parameterization—making it versatile in handling both regional and global weather patterns.

### Strengths
### Originality and significance 
Prithvi WxC pushes the foundation model concept in atmospheric science further by expanding beyond just forecasting— seen in earlier models foundation model Aurora. Its architecture and training approach enable it to tackle a variety of downstream tasks, such as downscaling and parameterization, making it a valuable tool for both short-term weather predictions and long-term climate modeling. With its modular design, Prithvi WxC can be adapted flexibly to new tasks that combines AI with physical climate models.

       
### Quality
The authors have thoroughly evaluated Prithvi WxC across a range of tasks, including zero-shot reconstruction, downscaling, and extreme event forecasting. The extensive validations across different downstream tasks support Prithvi WxC’s adaptability and effectiveness in diverse weather and climate applications.



### Clarity and open research
The paper is organized in a clear, logical flow, moving smoothly from motivation and background to model architecture, objectives, and results. Key ideas, like the mixed masking and forecasting objective, are presented in a way that makes the technical contributions accessible to both AI and climate science audiences. Code and comprehensive supplementary materials are provided.

### Weaknesses
### Reliance on a single reanalysis dataset (MERRA-2)
MERRA-2, with its relatively lower spatial resolution, is not commonly used in AI-driven weather and climate research, where higher-resolution datasets like ERA-5 are preferred for their superior predictive accuracy. The authors themselves acknowledge this limitation, citing the weaker performance of Prithvi WxC in hurricane track forecasting compared to the ERA-5-trained FourCastNet, attributing this discrepancy to MERRA-2's lower spatial resolution.      

Prithvi WxC’s exclusive training on the MERRA-2 reanalysis dataset raises questions about whether it truly qualifies as a foundation model. The narrower training base implies that the model may be learning a representation more specific to MERRA-2 characteristics, along with its biases and errors, rather than capturing a broader, more generalized understanding of weather and climate dynamics. By contrast, Aurora foundation model are pretrained on six diverse weather and climate datasets, including ERA-5, CMCC, IFS-HR, HRES Forecast, GFS Analysis, and GFS Forecasts, which span various sources like forecasts, analyses, reanalyses, and climate simulations [1]. This multi-source approach ensures a broader, more representative foundation that enhances versatility across diverse applications.Prithvi WxC would benefit from a similar multi-dataset training approach to strengthen its robustness and generalizability, which would then qualify it as a foundation model.

[1] Bodnar, C., Bruinsma, W. P., Lucic, A., Stanley, M., Brandstetter, J., Garvan, P., ... & Perdikaris, P. (2024). Aurora: A foundation model of the atmosphere. arXiv preprint arXiv:2405.13063.

### Ablation Study
While the authors propose a novel objective function, they only hypothetically attribute Prithvi WxC's strong short-term forecasting performance to its masking objective, without providing empirical evidence. This lack of testing weakens the claims about the model’s unique architecture and objective function. The observed performance could be influenced by several factors: the mixed objective itself, specific network structures or attention mechanisms, and choices made in the pretraining setup.

Without ablation experiments, the paper's assertions about the effectiveness of these innovations remain speculative, leaving readers uncertain about the impact of each component. An ablation study could isolate the contributions of these elements and would strengthen the paper by making its claims more concrete and providing clearer insights into Prithvi WxC’s architectural and training contributions.

### Questions
### Minor issue:  Weak baselines for downscaling
The comparison primarily involves interpolation-based methods and does not consider more advanced, AI-driven downscaling models or domain specific statistical/dynamical downscaling.

### Soundness
2

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces a new foundation model, Prithvi WxC, for atmospheric modeling applications in weather and climate.  Prithvi WxC was trained on 3-hourly data from 1980 to 2019 from the MERRA-2 reanalysis dataset based on a masked reconstruction/forecasting pre-training objective. 
The model follows a transformer-based encoder-decoder architecture inspired by Hiera and MaxViT.
Afterward, the model is fine-tuned for various downstream tasks: Medium-range weather forecasting, global and regional downscaling, and learning a gravity wave flux parametrization. These tasks have different sets of spatial resolutions, variables, and datasets, showing the flexibility of the foundation model.

### Strengths
1. Prithvi WxC is quite flexible as it can be used for a broad range of downstream applications, as convincingly shown in the experiments. 
2. The method contains original ideas such as the pre-training objective and using climatology-derived anomalies as targets, which I found interesting to read about.
3. The paper is generally clearly written and easy to read.

### Weaknesses
1. The paper falls short of establishing a compelling case for Prithvi WxC as a foundation model for weather or climate. The practical significance and advantages of this approach remain inadequately demonstrated:

a.) While foundation models typically excel at zero-shot performance and data-efficient fine-tuning across diverse tasks, the evidence presented for Prithvi WxC's capabilities in these areas is not convincing. Baselines for the non-forecasting experiments are either very weak (interpolation-based downscaling) or non-existent (gravity wave experiments). Some highly relevant and simple baselines are: 
- How much worse(?) does Prithvi WxC perform on these tasks if you omit the pre-training stage (i.e. initialize with random weights instead of the frozen pre-trained ones, and train all parameters jointly from scratch on the tasks)? 
- How about completely removing the pre-trained transformer backbone (i.e. removing the Prithvi WxC block from Figures 12 & 13)? 
- For the latter, it would be also good to run an experiment where you replace the pre-trained Prithvi WxC backbone with some "lightweight" blocks (e.g. a (deeper) U-Net), trained in a task-specific way from scratch, to account for the huge difference in parameter counts if you completely remove Prithvi WxC. 

These ablations would immensely help in understanding how useful the pre-training stage is for these downstream applications (e.g. does using pre-trained Prithvi WxC improve performance over such simple baselines? Is it more data-efficient?). Besides, otherwise, it is hard to see evidence for the claim in the conclusion that *"Instead of building task-specific ML-models from scratch, these pretrained encoders can be used to develop more precise data-driven models of atmospheric processes"*.

b.) No ablations are included. I understand that training such a huge model is expensive but having a few ablations would have been very appreciated (perhaps, with a smaller-scale version of the model). For example:

- How crucial is it to predict climatology-normalized targets as opposed to normal per-variable means/stds? 
- What's the forecasting performance of Prithvi WxC after the first pre-training phase?
- How important is local vs global masking? What about the masking rates?
- What's the line of thought behind randomizing the distance between input timesteps? Can the model only use one input timestep? I presume this is possible by masking the corresponding snapshot by 100%, but no experiments with this setting are shown. 

c.) The weather forecasting results seem lukewarm, albeit it is hard to judge because the comparison is not apples-to-apples.
- Prithvi WxC is trained and evaluated on Merra-2. The baselines are evaluated on ERA5. These reanalysis datasets have different spatial resolutions. The evaluation years seem to be different too (correct me if I'm wrong). It would help to fix this mismatch. For example,  given the foundational nature of Prithvi WxC... why not fine-tune it on ERA5 directly? Showing that it can be competitive to these baselines in an apples-to-apples comparison would be a very strong result.
- Based on the mismatched comparison, Prithvi WxC seems to be competitive on 6h to 12h forecasts but it's quite notable that its performance implodes compared to the baselines for longer lead times. It is very unclear why. I wouldn't necessarily expect this version of Prithvi WxC to be state-of-the-art, but the performance does seem underwhelming. Especially given that the authors did "several things" to tune these results (i.e. a second forecasting-specific pre-training stage and autoregressive rollout fine-tuning).
- The hurricane evaluation includes hurricanes from 2017 to 2023. This seems to overlap with the training data period (up to 2019). 
- Either Figure 6 or its analysis in the main body of the text (lines 251-253) is wrong because I see all of the three models do best on exactly one of the three RMSE figures.
- For the hurricane forecasting experiments, I would appreciate a comparison to the state-of-the-art models included in the weather forecasting experiments (e.g. GraphCast) which have shown to be better than FourcastNet.

d.) The downscaling problem setup is artificial. Downscaling coarsened of existing reanalysis/model outputs is not of much use in practice. A realistic and important downscaling application, as discussed in the Appendix, would be to downscale coarse-resolution model outputs to high-resolution outputs (either of a different model, observations, or the same model run at higher resolution).
 
e.) The climate model parameterization experiments should be more carefully interpreted. 
- The model predicts outputs that are normalized by the 1980-2019 climatology. Unfortunately, decadal or centennial simulations of the future under a changing climate are inherently a non-stationary problem. It is highly unclear if Prithvi WxC would remain stable, let alone effective, under this highly relevant use case. This is particularly so as the in-the-loop (coupled to a running climate model) stability of ML-based climate model parameterizations is a well-known issue.
- The selling point for ML-based emulators of climate model parametrizations is often their computational cheapness. Thus, the runtime of Prithvi WxC should be discussed. Given the large parameter count of Prithvi WxC it might be important to note its runtime as a limitation for these kinds of applications.
- Line 461 claims that Prithvi WxC "outperforms" task-specific baselines but no baselines whatsoever are included in the manuscript for this experiment.
- Are the inputs a global map? I am not familiar with gravity waves, but I believe that most physics parameterizations in climate models are modeled column-wise (i.e. across atmospheric height but ignoring lat/lon interactions). This is surely a simplification of these parameterizations, but it seems to indicate that they're highly local problems. What's the motivation for using global context then?
- The end of the section should be worded more carefully, clearly stating the aforementioned limitations.

f.) No scaling experiments are included. Thus, it is unclear how important its 2.3 billion parameter size is, how well the model scales, and how its size impacts performance on the downstream applications. Besides, vision and language models are usually released with multiple model sizes that cover different use cases (e.g. balancing inference speed with accuracy). It would be really useful to get these (and carefully compare them) for Prithvi WxC.

2. Related work is insufficiently discussed. Please include an explicit section discussing it, focusing on:
- Carefully comparing similarities/differences to existing weather foundation models (e.g. architectures, pre-training objectives, downstream applications etc.). Besides, ClimaX is not properly discussed in the paper. Given that it's also a transformer-based foundation model, validated on forecasting, downscaling, and climate emulation, it is very important to include it in the comparison. 
- Similarly, please discuss how exactly the masking technique in this paper relates to the ones proposed in Vandal et al. and McNally et al..
- Carefully discuss how the architecture is derived from Hiera and/or MaxViT (and other papers of which components were derived, if any).

3. While the authors transparently discuss some issues/limitations with their experiments (e.g. the evaluation data mismatches), it would be nice to also include an explicit paragraph or section on this (and include aforementioned things like the issues with the climate model parameterization experiments).

Minor:
- Can you properly discuss, and include a reference to, what a Swin-shift is?
- Similarly, for the "pixel shuffle layers"
- Line 39: Pangu -> Pangu-Weather
- Line 48: Nowcasting should be lower-case
- Equation 1: Consider reformulating this as an objective/loss function.
- Also Eq. 1: What is $\hat{X}_t$? What is $\sigma_C$?
- Line 93: $\sigma^2_C = \sigma^2_C(X_t - C_t)$ doesn't make sense to me.
- Line 104: *" same 20 year period that we used for pretraining."* .... Do you mean 40 year period? If not, which 20-year period from the 40-year training period did you use?
- Line 157: Multiple symbols are undefined (e.g. $V_S$).
- Line 169: It's not entirely clear what "alternates" means in this context.
- Line 429: "baseline"... do you mean Prithvi WxC? 
- Line 507: "improved"... improved compared to what?
- Figure 12: Do you mean 'downscale' on the right "upscale" block?
- Sections D. 2.3 and D.2.4 in the appendix are literal copies of the corresponding paragraphs on pages 8 and 9. Please remove.

### Questions
Major: 
- Do you compute area-weighted RMSEs? If not, I strongly encourage fixing this (especially for the weather forecasting experiment; see e.g. Weatherbench2).
- For the fine-tuning experiments, do you start them based on the weights resulting from the first or second pre-training stage? If the former, then the second pre-training seems to also be a form of fine-tuning and I would suggest avoiding using the term "zero-shot" for reporting the forecasting results.
- From looking at Figure 3, I don't think that I agree with the author's interpretation that *"It is interesting that reconstruction performance is relatively little affected by lead time at the lower end of masking ratios"*. There's a clear cap between "0h, global" and "6h, global", especially for the lower end of masking ratios. Do I miss something?
- Can you include downscaling results on more variables than only T2M?

Minor:
- What's the difference between the encoder and decoder blocks? Fig. 1 suggests these are identical... Is the difference some densification by the "reconstruct batch" module in Fig. 1? If so, can you explain this more and make it clearer?
- Line 174: "reduce the masking ratio to 50%"... is this a typo (you use the same rate for the first stage)? What's correct?
- Why is there no reference line in figure 5b)?
- Please define what's meant by spatial and temporal RMSEs.
- Can you include snapshots like Fig. 9 but at the native temporal resolution used for prediction (i.e. not a monthly mean)?
- How does using different patch sizes (larger than the used size of 1) impact downscaling performance? 
- Why do you call the model Prithvi WxC?

### Soundness
2

### Presentation
3

### Contribution
3

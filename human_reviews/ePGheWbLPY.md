# Gridded Transformer Neural Processes for Large Unstructured Spatio-Temporal Data

- Decision: Reject
- Scores: 6, 8, 5, 3

## Abstract
Many important problems require modelling large-scale spatio-temporal datasets, with one prevalent example being weather forecasting. Recently, transformer-based approaches have shown great promise in a range of weather forecasting problems. However, these have mostly focused on gridded data sources, neglecting the wealth of unstructured, off-the-grid data from observational measurements such as those at weather stations. A promising family of models suitable for such tasks are neural processes (NPs), notably the family of transformer neural processes (TNPs). Although TNPs have shown promise on small spatio-temporal datasets, they are unable to scale to the quantities of data used by state-of-the-art weather and climate models. This limitation stems from their lack of efficient attention mechanisms. We address this shortcoming through the introduction of gridded pseudo-token TNPs which employ specialised encoders and decoders to handle unstructured observations and utilise a processor containing gridded pseudo-tokens that leverage efficient attention mechanisms. Our method consistently outperforms a range of strong baselines on various synthetic and real-world regression tasks involving large-scale data, while maintaining competitive computational efficiency. The real-life experiments are performed on weather data, demonstrating the potential of our approach to bring performance and computational benefits when applied at scale in a weather modelling pipeline.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
Transformers have shown promising results in weather forecasting but are limited to gridded structured data and neglect large amounts of unstructured observational data. 
Transformer Neural Processes (TNPs) are more flexible, capable of handling both structured and unstructured data, but they struggle with scalability. 
To address this issue, this paper introduces gridded pseudo-token TNPs.
Gridded TNPs consists of an encoder to convert the unstructured observation data to a structured grid via cross attention, a transformer (i.e., Vision Transformer [ViT] or Swin Transformer)-based processor to generate gridded token prediction, and a decoder with k-nearest-neighbor cross attention to map the predicted token back to the unstructured data at arbitrary locations.
The paper evaluates the methods on a few benchmarks, comparing the pseudo-token grid encoder to the kernel interpolation approaches and ViT to Swin Transformer.

### Strengths
The proposed gridded pseudo-token encoder and decoder with k-nearest-neighbors via cross-attention provided a flexible and elegant way to deal with both unstructured and structured data. The paper is well written, and the method is clearly presented.

### Weaknesses
1. The experiment results are not analyzed in detail enough to substantiate the claims. The reviewer had a difficult time to link the very general discussions in section 5.4 to the experiments in sections 5.1-5.3. Adding a few sentences on the observations/conclusions of each experiment would be helpful.
2. The claim that introducing ViT and Swin Transformer into the transformer backbone as an efficient mechanism is a main contribution feels somewhat weak.
2. Typos: 
	1) In Fig. 2, the “{(x_{c,n},z_{c,n})}_n=1^N_c” should be “{(z_{c,n})}_n=1^N_c”?
	2) Line 344 on page 7, “e_s(x_{c,n,s},x_{c,n,s})” should be “e_s(x_{c,n,s},y_{c,n,s})”?

### Questions
1. GNNs are another type of commonly used approaches for handling unstructured data? Could the authors comment on them?
2. Compared to KI-GE, PT-GE appears to achieve better accuracy but requires a higher FPT. In the ViTNP cases in Table 1, the FPTs are 215 vs. 171 for PT-GE and KI-GE, respectively. Could the authors provide an explanation for this difference in time cost?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
4

### Summary
The paper proposes Gridded Transformer Neural Processes (Gridded TNPs), an extension of TNPs that scales better to large-scale datasets. The idea is to use pseudo tokens and cross-attention to bring unstructured, point-based data to a grid before further processing by efficient on-grid architectures such as ViTs and Swin Transformers, and finally decoding to off-grid predictions via cross-attention. Different variants of Gridded TNPs were also introduced with different grid encoding methods and processor architectures. Gridded TNPs outperform the baselines on synthetic regression and station-based weather forecasting.

### Strengths
Overall, I think the ability to blend off-grid and on-grid data of Gridded TNPs contributes well to the spatiotemporal modeling community. I like the simplicity of the proposed method and the clear writing of the paper. The experiments, even though are still at small scales, are reasonable for an academic setting, and sufficiently show the good performance of Gridded TNPs.

### Weaknesses
I do not see any critical weakness in the paper at the moment, but would like some clarifications and more analyses of the design choices in the paper. Specifically:
- What are the queries, keys, and values in the cross-attention operation in the grid encoder? What I understand from the paper is you use the learnable pseudo tokens as queries and embeddings of context points as keys and values. If this is true, do you use any positional encoding for the pseudo tokens?
- The paper said k=1 for the grid encoder is sufficient. Does this mean each pseudo token only attends to one context point that is closest to its grid location? What is the difference between this and simply picking the closest context point?
- In the grid decoder, where do the target tokens Z_t come from? Are they embeddings of the target inputs x's?
- I'd like to see more ablation studies about choosing k in both the encoder and decoder across different datasets and tasks.

----- POST REBUTTAL -----

The authors have addressed my main concerns so I'm raising my score to 8.

### Questions
See above.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a Gridded Transformer Neural Process (G-TNP), which effectively maps unstructured spatiotemporal data onto a grid and incorporates an efficient attention mechanism, enhancing predictive performance on large-scale datasets.

### Strengths
The paper systematically introduces the G-TNPs model architecture, including the grid encoder, grid processor, and grid decoder, each accompanied by detailed mathematical descriptions and theoretical foundations.

### Weaknesses
1. I'm curious, have you considered using large-scale meteorological models like Pangu as a baseline for comparison?
2. Although an efficient attention mechanism has been introduced, the model's complexity might still be high, especially when handling ultra-large-scale datasets. Could the authors provide comprehensive data to clarify this?

### Questions
See Weaknesses.

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 4

### Rating
3

### Rating Number
3

### Confidence
4

### Summary
A gridded transformer architecture is introduced that builds on neural processes. The model builds on three stages. First, unstructured data is encoded into a grid, second the grid is processed with an efficient ViT or SwinTransformer architecture, and third the grid is decoded to provide predictions at arbitrary positions beyond the grid. A comparison to other deep learning methods for processing unstructured grids demonstrates the benefits of the introduced method.

### Strengths
_Originality:_ The presented approach is original by combining established methods, such as Transformer Neural Processes, with efficient attention mechanisms from ViT and SwinTransformer, to tackle a highly relevant real-world problem. It would be very interesting to see, how the proposed method compares to ERA5 data and to understand the biases this approach imposes. Coming up with a data driven ERA5 alternative that mitigates numerical model biases would be of extraordinary value.

_Quality:_ The consideration of uncertainties in the output is a great design choice for the task of working with atmospheric states. Figures are mostly well designed and support the messages conveyed in the manuscript. Honestly, I do not see the value of Figure 1, in particular, since Figure 2 outlines the same pipeline both more detailed and concrete.

_Clarity:_ The manuscript is well organized, structured, and written. A rich appendix provides details about the model design, and a supplementary material with code is provided.

### Weaknesses
Overall, the paper touches on many interesting aspects, but misses a central message and problem that is tackled. Advertising the model more towards an alternative of traditional data assimilation techniques via numerical weather prediction models could be an interesting focus.

_Significance:_ It is hard to assess, whether the introduced method offers advantages over traditional data assimilation methods. To this end, the manuscript reads like introducing a neural network for data driven data aggregation (i.e., an alternative to ERA5), which potentially would provide substantial value. I would therefore suggest a focus shift from minimizing forecast error, such as done by all these deep learning weather prediction models, to generating a coherent state of the atmosphere, which is much more novel. Comparing to ERA5 would be highly appreciated, even though one cannot expect to outperform this dataset yet. Reporting first steps would be very insightful, though, and of large interest for the community.

_Quality:_
- **Qual1** Apart from the positive quality aspects mentioned under "Strengths," the manuscript can be improved in terms of supporting claims. That is, neural processes are introduced as flexible to downstream tasks, such as forecasting, data fusion, data interpolation and data assimilation (ll. 60--61). However, the model is only evaluated in terms of diagnosis error but not on the quality of other tasks.
- **Qual2** The use of different encoders for each input modality (l. 342) sounds inspired by ClimaX, which would be fair to be referenced here.
- **Qual3** Due to scaling reasons, the introduced method is not compared to its predecessors ANP and TNP (l. 366). However, it would be valuable to understand how the introduced method performs on a small dataset, where ANP and TNP are applicable, to understand whether the decign choices made in the new model result in performance decrease, and if so, to what extent.
- **Qual4** The manuscript advertises the transition from gridded reanalysis dataset, such as ERA5, from numerical weather prediction models. Yet, the model depends on ERA5 as additional gridded input beyond the heterogeneously distributed weather station data. It is unclear, how the gridded data and the weather station data is integrated and why the gridded data is necressary.
- **Qual5** GraphCast and GenCast follow a similar processing pipeline, where data points are aggregated into a grid, then processes, and subsequently decoded. A comparison to the aggregation approaches in these methods would be insightful. Otherwise it is hard to assess the quality of the introduced method over well established approaches.
- **Qual6** Figure 14 is very hard to interpret. A colourmap as well as a difference plot to the ground truth would help substantially to assess the quality of the different models. Similarly, it is difficult to extract the content of Figure 16, which is too dense. Zooming into certain regions could help.

_Clarity:_ Readers might benefit if the prediction task in Section 5.2 and Section 5.3 was introduced as diagnosis, i.e., diagnosing from some input stations in time $t$ to other input stations in the same time step $t$, where no forecasting is involved.

### Questions
1. How is the latent grid structured? That is, what resolution is chosen and how are the grid points distributed over the sphere, e.g., equirectangularly? Have you considered distortions towards polar regions in ERA5 data? I could find some information in Table 4 of the Appendix, but could not see for which tasks these grids were chosen (i.e., global vs local diagnosis).
2. What effect does the size of the attention circle (or diameter) in the decoder have on the performance? Depending on the target variable, this might be a crucial hyperparameter. However, increasing the attention radius increases the computation time. Can you provide some information on the relevance of the size of the attention circle for different target variables (e.g., t2m vs. wind) and what computational cost the size of the decoder's attention circle imposes?

### Soundness
1

### Presentation
3

### Contribution
2

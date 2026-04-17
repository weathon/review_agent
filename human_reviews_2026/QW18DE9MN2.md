# BioAnalyst: A Foundation Model for Biodiversity

- Decision: Reject
- Scores: 2, 2, 2, 6

## Abstract
Multimodal Foundation Models (FMs) offer a path to learn general-purpose representations from heterogeneous ecological data, easily transferable to downstream tasks. However, practical biodiversity modelling remains fragmented; separate pipelines and models are built for each dataset and objective, which limits reuse across regions and taxa. In response, we present BioAnalyst, to our knowledge the first multimodal Foundation Model tailored to biodiversity analysis and conservation planning in Europe at $0.25^{\circ}$ spatial resolution targeting regional to national-scale applications. BioAnalyst employs a transformer-based architecture, pre-trained on extensive multimodal datasets that align species occurrence records with remote sensing indicators, climate and environmental variables. Post pre-training, the model is adapted via lightweight roll-out fine-tuning to a range of downstream tasks, including joint species distribution modelling, biodiversity dynamics and population trend forecasting. The model is evaluated on two representative downstream use cases: (i) joint species distribution modelling and with 500 vascular plant species (ii) monthly climate linear probing with temperature and precipitation data. Our findings show that BioAnalyst can provide a strong baseline both for biotic and abiotic tasks, acting as a macroecological simulator with a yearly forecasting horizon and monthly resolution, offering the first application of this type of modelling in the biodiversity domain.
We have open-sourced the model weights, training and fine-tuning pipelines to advance AI-driven ecological research.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this paper the authors propose BioAnalyst a  model presented as a forecasting foundation model for biodiversity in Europe at 0.25 degrees resolution.  More specifically, the model is trained with BioCube which encompasses 10 modalities including climate and species range data. The proposed architecture is a composed of a PerceiverIO encoder, a 3d swin transformer backbone and a PerceiverIO decoder. 
The model is  pre-trained on the task of forecasting at the next time step. Then the model is evaluated on downstream tasks including forecasting biodiversity dynamics, and species distributions.

### Strengths
This project tackles an important challenge and few studies have incorporated these many types of input for biodiversity monitoring applications. The paper is generally well structured too.

### Weaknesses
In general, the applications in ecology would benefit from being explained in a clearer way: 

- For example, the task could be better defined especially when referring to species distribution: there are many things that can be modelled in species distribution modelling (encounter rates, abundance, occurrence, occupancy). It is unclear what you are trying to model. Can you please clarify what is predicted in terms of species distribution? 

- I would like to understand how you think that a 28km resolution is what is needed for conservation decisions. I would perhaps be a bit more conservative about this statement.  It would be good for the authors to outline more precisely what kind of decisions a map at this scale could be helpful for. In practice for conservation decisions especially when looking at specific species (which is the use case the authors are highlighting), much higher resolution is needed.  I am also surprised this is the resolution that is chosen given the higher resolution of many of the products that are used (e.g. satellite imagery) especially in Europe. It seems like this alignment at 28km loses a lot of the fine-grainedness of the original raw data source. I understand that this might be the approach taken with BioCube but I wonder if then it is not the most suited dataset for this task.

Evaluation: 
- I am confused about the choice of looking at cumulative mean of distribution across 28 species distributions (Fig 3). This actually doesn't reflect error. Instead you should show the error (you can average it across space and species if you want) but otherwise it can be very misleading. also, the claim "The predictions closely follow the data trend, " doesn't seem very well supported.

### Questions
In addition to asking for clarification about the points in the weaknesses, I have the following questions: 

- the model is pre-trained with prediction one time step ahead from the past. Hae the authors tried other number of steps ? Or random number of steps (with additional conditioning on the number of steps to predict ahead) ? Why would a task of interpolation not be better suited?  How well is the model able to predict on longer time frames? 

Evaluation: 
- sorensen similarity map in Fig 4: I almost seems like the species match is correlated with data sampling (if we think about number of observations in general). Could the authors elaborate on that? Are there spatial biases in the species data that is used to validate these maps? 
- Fig 6: I might have misunderstood but it seems that in the ground truth not all data is observed. How do you know that Aurora overpredicts if there are only a few cells with which to validate the map?

- On another note, this is  more of a comment about something that could be highlighted in the discussion than a weakness, but the RLI is not necessarily suited for this task. I would be curious to see if there are ablations of the different modalities used for this model. Indeed, the problem with the IUCN red list is that for a given species, its assessment is updated every 10 years. 

- predicting at the next time step: how do the authors think this can be suited for capturing things like seasonal patterns? 
-  C.4.2.: I think this would benefiot from clarification about how the observations are aggreggated, and how "The model preserves the full spatial resolution of the data" since the observations of geolifeclef are point data? 

Other comments: please cite the references with parentheses, it makes it difficult to parse the sentences otherwise. 

I am open to revising my score, but in the current state, the definition of the downstream tasks is quite unclear to me as well as the pathway to impact.

### Soundness
1

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This article describe a foundation model pre-trained on the BioCube dataset. This dataset aggregates a diverse set of data sources into a geospatial data cube at a 0.25 deg resolution. These data sources cover different aspects of the environment that are relevant to biodiversity, including soil and climate variables or species observations.
The model is based on a PerceiverIO architecture for the encoder and decoder with a Swin Transformer as backbone, and the pre-training objective is a MAE.
Two downstream tasks are used for evaluation: species distribution mapping and forecasting based on GeoLifeCLEF2024 and monthly climate recovery based on CHELSA.

### Strengths
- The motivation is clear: there is currently no foundation model that has been pretrained on such a variety of variables relevant to biodiversity-related tasks.

### Weaknesses
1. I find the evaluation to be far from enough to drive the message of the paper. With only two tasks, one of them not directly related to biodiversity, it is hard to convince the readers that the proposed model actually serves as a foundation model. In addition, only one competing method is shown: Aurora, which is intended as a climate foundation model and is thus not a great fit for the species distribution task (I couldn’t find a comparison for the climate recovery task, where Aurora would probably do a good job). I would expect a comparison against several purely supervised baselines (with the same architecture and also with simpler architectures) and against more suitable competitors, including Earth observation-focused foundation models and pre-computed embeddings (such as AlphaEarth).
2. Even after having a look at the suppl. material and the BioCube paper, I struggle to understand the nature of the dataset used. For instance, Table 2 displays a list of species and their total number of occurrences. Does this mean that only occurrences of these species are used as input? Are these counts simply aggregated over 0.25 deg grid? In Eq. (4) we see how the input to the model is put together, but I find it confusing. Do all variables have 10 levels? (not just pressure?). If there are two nested sums, this means all variables and level are summed together? I have spent some time on this (again, both on this paper and the original BioCube), and I still don’t see how exactly the input data looks like.
3. I am unable to understand the nature of the used model. The description clearly states that encoder and decoder are based on cross-attention (as done in PerciverIO), and that, in between, there is a Swin Transformer. However, Swin Transformers are specifically designed for spatially arranged representations (such as image tokens), while the representation provided by a PerceiverIO encoder lacks any notion of spatial arrangement. How are both put together?
4. In connection to point 1, I felt there is a lack of discussion of what types of downstream tasks the model would be a good fit for. After all, the spatial resolution would make it unsuitable for many local downstream applications.

### Questions
I would like to read an answer to the questions formulated in weakness 2 and 3:
- What is the dimensionality of each data modality? How are the different modalities built?
- How are the PerceiverIO and Swin Transformer architectures put together? (given that the first provides non-spatial representations and the second requires explicitly spatial ones).
Other than this, I believe there is still a lot to do wrt weakness 1 in order to sustain the claim that the proposed model is a foundation model.

### Soundness
1

### Presentation
1

### Contribution
1

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper introduces BioAnalyst, a multimodal foundation model for biodiversity analysis and ecological forecasting.
The model integrates ten heterogeneous data modalities — including climate, vegetation, soil, and species occurrence variables — through a hybrid architecture combining a Perceiver IO encoder–decoder with a 3D Swin Transformer backbone.
BioAnalyst is pretrained on a 20-year subset (2000–2020) of the BioCube dataset at 0.25° spatial resolution over Europe and is fine-tuned for three downstream tasks: (1) biodiversity rollout forecasting, (2) species distribution modeling, and (3) climate variable reconstruction (linear probing).
The authors claim that BioAnalyst outperforms Microsoft’s Aurora climate foundation model, especially in data-scarce scenarios, and publicly release model weights and code.

### Strengths
- Timely and societally valuable problem. Bringing foundation-model methodology to biodiversity and conservation is meaningful and impactful for environmental science.

- Ambitious multimodal integration. The model aligns diverse ecological variables within a single latent representation, a nontrivial data-engineering achievement.

- Transparent and reproducible. Code, model weights, and detailed appendices are provided.

- Clear writing and motivation. The introduction and related-work sections convincingly position biodiversity modeling as an under-served domain for AI FMs.

### Weaknesses
- Engineering integration rather than scientific novelty.
The architecture is an adaptation of existing components (Perceiver IO, Swin Transformer) with minor modifications and a temporal-difference loss.
While this is solid engineering, it lacks new methodological ideas or theoretical insights expected at ICLR.

- Over-reliance on outperforming Aurora.
The only strong empirical claim is that BioAnalyst “outperforms” Aurora on a few metrics.
However, Aurora is primarily a climate model, not a biodiversity model, and its temporal/spatial resolutions differ.
Therefore, this comparison is not fully fair, and exceeding it on select metrics does not establish a significant scientific contribution.
In short, beating Aurora is not, by itself, enough to justify an ICLR paper.

- Inconsistent metric reporting.
The authors report 𝑅2 for the abiotic task (where results are strong) but omit it for the biotic fine-tuning task, where performance is less clear.
This selective reporting weakens confidence in the claim of superiority.

- Limited empirical rigor.
No ablations on architecture or modalities, and no additional baselines beyond Aurora.
The F1 difference (0.996 vs 0.994) is negligible, and uncertainty quantification is missing.

- Narrow evaluation scope.
The dataset covers only terrestrial Europe at 0.25° resolution, and the model is not tested globally or across unseen taxa.
This makes it difficult to assess generalization and robustness.

- Scientific framing.
The paper presents a well-engineered pipeline but does not reveal new scientific understanding or modeling principle.
The contribution lies in integration and system building rather than discovery.

### Questions
- Report consistent metrics (including 𝑅2, correlation, and uncertainty) for both biotic and abiotic tasks.

- Include additional baselines and ablation studies, e.g., single-modality or reduced-modality versions.

- Clarify the ecological interpretation of what BioAnalyst learns (e.g., which cross-modal relationships drive predictions).

- Temper the Aurora comparison, acknowledging task mismatches and limited evidence.

- Consider submission to a domain-specific venue such as Environmental Modelling & Software, Ecological Informatics, or Nature Scientific Data, where the work’s environmental and integrative value would be more impactful and appropriately appreciated.

### Soundness
2

### Presentation
2

### Contribution
3

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
The paper presents bioanalyst, a transformer-based, multimodal foundation model for biodiversity. It integrates diverse geospatial and ecological data and reports strong results on forecasting tasks.

### Strengths
- integrates diverse data types (remote sensing, climate, soils, species, etc.).
    
- code is available and opensourced
    
- comparison with microsoft’s aurora gives an initial external reference point.
    
- generally well-written with detailed implementation details.

### Weaknesses
- Only two downstream tasks are shown (species forecasting; climate probing), with limited baselines (Aurora only).
    
- Can include task-specific baselines.
    
- why a swin transformer? please justify the backbone choice.
    
- How are the encoders-decoders trained? Together with the whole model or are they pretrained beforehand?
    
- There is no uncertainty quantifcation discussed in the work. It would be good to have some experiments around that.
    
- Has the model been trained to predict x_{t+1} directly (next-step forecasting)? how sensitive are results to this choice? What if it is trained with x_{t+2} or if it is varied during training?
    
- The paper would benefit with some analysis on which features impact the model more? Perhaps a qualitative analysis.
-  The work would further benefit from more out of distribution experiments and zershot analysis being a foundation model

### Questions
- What is the contribution of each modality group (climate/soil/land/NDVI/species) to downstream performance? 
    
- why a swin transformer over alternatives for spatiotemporal data? 
    
- can you show zero-shot performance on out-of-distribution regions/years and compare against aurora
    
- do the features/modalities share the same grid/temporal resolution? if not, what resampling/normalization/alignment steps are used before fusion?
    
- which datasets were used for fine-tuning each downstream task? is there any species/time overlap with pretraining that could bias results?
    
- Please clarify the rollout loss.

### Soundness
3

### Presentation
3

### Contribution
3

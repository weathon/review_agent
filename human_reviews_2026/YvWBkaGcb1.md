# MoveGPT: Scaling Mobility Foundation Models with Spatially-Aware Mixture of Experts

- Avg Score: 3.50
- Decision: Reject
- Scores: 4, 2, 2, 6

## Abstract
The success of foundation models in language has inspired a new wave of general-purpose models for human mobility.  However, existing approaches struggle to scale effectively due to two fundamental limitations: a failure to use meaningful basic units to represent movement, and an inability to capture the vast diversity of patterns found in large-scale data. In this work, we develop MoveGPT, a large-scale foundation model specifically architected to overcome these barriers. MoveGPT is built upon two key innovations: (1) a unified location encoder that maps geographically disjoint locations into a shared semantic space, enabling pre-training on a billion scale; and (2) a Spatially-Aware Mixture-of-Experts Transformer that develops specialized experts to efficiently capture diverse mobility patterns. 
Pre-trained on billion-scale datasets,  MoveGPT establishes a new state-of-the-art across a wide range of downstream tasks, achieving performance gains of up to 35\% on average. It also demonstrates strong generalization capabilities to unseen cities. Crucially, our work provides empirical evidence of scaling ability in human mobility, validating a clear path toward building increasingly capable foundation models in this domain. The source code and pre-trained models for MoveGPT are publicly available at: https://anonymous.4open.science/r/MoveGPT-FC72/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces MoveGPT, a large-scale foundation model for human mobility modeling that aims to generalize across multiple cities. The proposed framework combines two key components: (1) a unified location encoder that projects heterogeneous geographic and functional features (e.g., POI distribution, coordinates, popularity) into a shared semantic space; and (2) a Spatially-Aware Mixture-of-Experts (SAMoE) Transformer, which allocates trajectory segments to specialized experts (POI, Geo, Pop) via a spatial–temporal router (STAR). The model is pre-trained on over one billion mobility trajectories from up to 16 cities and evaluated across tasks such as next-location prediction, long-term forecasting, generation, and anomaly detection. Empirical results show strong improvements over prior baselines and good transferability to unseen cities with limited fine-tuning data.

### Strengths
1. The paper makes an interesting and timely contribution to the emerging area of mobility foundation models. It presents a large-scale pre-trained model trained on trajectories from 16 cities, demonstrating the feasibility of scaling human mobility modeling to a global level. The dataset size and coverage are impressive and provide a strong empirical foundation for large-scale pretraining.

2. The paper is easy to follow. The motivation, methodology, and experimental setup are logically organized.

3. The paper evaluates across multiple downstream tasks and includes scaling analyses with respect to data volume, model size, and city diversity. The multi-city pretraining and fine-tuning experiments convincingly demonstrate generalization potential.

### Weaknesses
1. The design of expert router is questionable. The STAR router is described as a binary selector between spatial and temporal gating, yet it is unclear why a binary mechanism should be used when the model processes three distinct trajectory feature types (POI, coordinates, and popularity). Using either spatial or temporal representations in the MoE process doesn’t make sense to me.

2. Although the model is termed a Spatially-Aware Mixture-of-Experts, the spatial awareness appears limited to the separation of trajectory features (coordinates, popularity, and category). The paper does not clearly explain how this mechanism captures spatial dependencies beyond feature partitioning, making the “spatially-aware” characterization somewhat unconvincing.

3. The paper claims that geographical coordinates are encoded using a shared MLP across all cities. However, since each city has vastly different coordinate ranges and spatial distributions, applying a single encoder directly across cities is questionable. Without normalization strategies that preserve inter-city spatial semantics, it is difficult to believe that this encoding process can be universally applicable.

4. The similarity-based next-location retrieval is essentially the same as token retrieval for generation process in LLMs. However, removing the softmax operation from the output distribution, as implied by Equation (11), raises concerns about whether the resulting scores still represent valid log probabilities.

5. Although the paper claims to release the resources, the referenced datasets do not appear to be publicly available. This restricts the reproducibility of the reported results. 

6. The Spatial-aware MoE architecture introduces multiple experts, routers, and feature retrieval modules, which likely increase computational overhead. However, there is no discussion of training time, memory footprint, or inference latency—factors critical for real-world deployment.

### Questions
1. The difference between the vector form $\mathbf{g}_i$ and the scalar $g_i$ in Equation (7) is unclear. How are these variables related, and what exactly is the role of each within the MoE computation? The overall formulation of the mixture-of-experts component needs clearer mathematical explanation.

2. Based on Equation 10, how can we make sure that Equation 11 is valid? 

3. How is the “next time” (timestamp) predicted in the model? The paper mentions temporal embeddings, but it is not clear how these embeddings are used for time prediction or how they interact with the autoregressive prediction process.

### Soundness
2

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
5

### Summary
This paper introduces MoveGPT, a large-scale foundation model for human mobility research, aiming to move the field beyond fragmented, task-specific approaches toward general-purpose, foundational systems. The authors propose: 1. A unified location encoder that maps geographically disjoint locations into a shared semantic space, enabling global-scale pretraining; 2. A Spatially-Aware Mixture-of-Experts (MoE) Transformer that leverages specialized experts to efficiently capture diverse mobility patterns. Pretrained on billion-scale datasets, MoveGPT achieves new state-of-the-art results across a wide range of downstream tasks, with average performance improvements of up to 35%.

### Strengths
1. The paper introduces the idea of applying Mixture-of-Experts (MoE) to the domain of mobility foundation models, which is relatively novel.

2. The authors conduct extensive experiments, demonstrating SOTA performance on multiple benchmarks.

### Weaknesses
1. Overclaiming novelty. The authors repeatedly emphasize their unified location encoder, but similar ideas have already appeared in prior works such as GeoCLIP [1], which also trained a unified location encoder and further extended it to downstream models like UrbanVLP [2]. MoveGPT’s encoder essentially encodes GPS coordinates with auxiliary features such as POIs, which is not a fundamentally new innovation.

2. Manual expert definition in MoE. In large language models, MoE architectures typically allow the model to autonomously learn expert specialization. In contrast, the authors manually define expert categories, which restricts scalability and partially contradicts the spirit of a general foundation model.

3. Incorrect claim about LLM architecture. In line 263, the authors claim that traditional methods use an MLP to predict the output distribution. However, most large models directly employ a linear projection on the final hidden representation, without an additional MLP.

4. Writing and formatting issues. The paper suffers from several writing problems: for example, in line 57, citations are not enclosed in parentheses; in line 200, terms such as POI and GEO are formatted inconsistently with standard academic conventions; and in line 261, there is an empty parenthesis. These suggest that the manuscript was prepared in haste.

5. Code availability mismatch. The abstract claims that both source code and pretrained models are publicly available in the anonymized repository. However, only the code is provided, while no pretrained models are included.

[1] Vivanco Cepeda, Vicente, Gaurav Kumar Nayak, and Mubarak Shah. "Geoclip: Clip-inspired alignment between locations and images for effective worldwide geo-localization." Advances in Neural Information Processing Systems 36 (2023): 8690-8701.

[2] Hao, Xixuan, et al. "Urbanvlp: Multi-granularity vision-language pretraining for urban socioeconomic indicator prediction." Proceedings of the AAAI Conference on Artificial Intelligence. Vol. 39. No. 27. 2025.

### Questions
1. Please clarify and respond to the identified weaknesses.

2. Please elaborate on the motivation for adopting a Wide & Deep network. What specific gap does it address in this context? Why is this considered an innovation? Have similar architectures not been explored in previous works on mobility modeling or foundation models?

### Soundness
3

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
In this paper, the authors have introduced a foundation model, named MoveGPT, that is able to tackle a diverse range of downstream human mobility tasks, such as next-location prediction, long-term prediction, conditional/unconditional generation, and anomaly detection. 
While the ideas behind the model are of interest, timely, and the challenges solved are socially relevant, the paper presents several limitations (see weaknesses below).

### Strengths
The three main strengths are the following ones:

Strength 1: The paper tackles problems that are extremely relevant for society. For instance, human mobility is known to be linked with disease diffusion, pollution, economic growth, and many other factors.

Strength 2: The model is carefully designed, and the idea of having a SAMoE handling different aspects of a city and human behaviors is appealing and well presented.

Strength 3: The model is evaluated on a diverse range of several different downstream tasks, covering most of the challenges related to human mobility. Tasks have been carefully selected.

### Weaknesses
The main weaknesses of the paper are the following ones:

Weaknesses 1: Many relevant pieces of information are not present in the main text and are presented only in the appendix, making the paper inconsistent and difficult to evaluate by looking only at the main content (see review policies). For instance, datasets used for training and evaluation are never presented in the main text. Additionally, datasets bring with them self-selection biases that, in cases like mobility, should not be underestimated. Overall, a proper description of the dataset is needed, and the limits of the dataset that may affect performance should be listed in a discussion/conclusion section properly.

Weaknesses 2: The authors boldly claim to have a model that can generalize (e.g., global-scale representation), but it seems that data for training and testing are US-only. However, it is well-known that the human mobility also has a cultural component (e.g., people in Spain go for dinner at 9:30/10 PM, while in Germany at 6/6:30, just to give some examples). If the datasets available do not allow for testing this global-scale representation, then it would be better to make less bold statements (e.g., a US representation that we believe may apply globally) or completely avoid them.

Weaknesses 3: The authors claim that a limit of other studies until now is that “they rely on inconsistent basic units to represent movements, from raw GPS coordinates to non-transferable Zone IDs.” Subsequently, in the definition of trajectory, they claim they map stay locations (i.e., GPS points) into cells of a tessellation. Why should this mapping be more transferable than others? To the best of my knowledge, some of the references the authors cite in the introduction rely on H3 grid mapping, which is exactly a cell with a unique ID that is non-overlapping with others. I am confused.

Weaknesses 4: Regarding this mapping with a cell, it is very well known that, in mobility, the results depend significantly on the size of the cells used. Also, intuitively, it makes sense. If you divide, for example, New York City into 10 cells of 20 km square or thousands of cells of 100 meters square, the complexity of the problem is tremendously different. However, the size of the cell adopted by the authors is never specified. In addition, they never report how cities are divided into cells and how cell IDs are assigned.

Weaknesses 5: When the authors present the unified encoder, what do they mean with “patterns are also influenced by broader social dynamics”? Does it mean there is also a role by collective behaviors, like in Calabrese, Francesco, Giusy Di Lorenzo, and Carlo Ratti. "Human mobility prediction based on individual and collective geographical preferences." or in Bontorin, Sebastiano, et al. "Mixing individual and collective behaviors to predict out-of-routine mobility."? In any case, some references would be appreciated to understand better what they mean by this.

Weaknesses 6: On line 186, it seems that the authors are encoding coordinates. However, in the definition, coordinates (which were also pointed out as a limitation of previous studies) are mapped into cells, so I am not sure about what the encoder is actually doing here. Similarly, when they define the semantic-aware sequence input, they embed POIs and the distribution of a certain zone. Also in this case, having an idea of how big a zone is is fundamental to understand the real performances of the model.

Weaknesses 7: When the authors describe the model training, they limit everything to next-location prediction. I was wondering if the authors also used other training methods like masking or other state-of-the-art techniques and, if not, if there is any specific reason.

### Questions
Some questions related to the above weaknesses: 

Who are the users in the datasets? What are their characteristics? What are the model's performances on people who are not represented in the dataset used?

How can one claim to have a model that can achieve a global-scale representation if it cannot be tested globally?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper presents a large-scale mobility foundation model, called MoveGPT, that aims to address two major obstacles faced by existing models when scaling. The paper point out that previous models have been limited by the lack of a unified unit of motion representation and the inability to efficiently capture the diversity of mobility patterns in large-scale data.MoveGPT overcomes these obstacles through two key innovations: (1) a unified location encoder that maps the geographic locations of different cities based on their functional, geographic, and social characteristics into a shared semantic space, thus enabling pre-training on a global scale for the first time; and (2) a Spatially Aware Mixture of Experts (SAMoE) Transformer architecture that allows for the efficient capture of diverse movement patterns in large-scale data. The model was pre-trained on a billion-scale dataset, achieving state-of-the-art performance on several downstream tasks (e.g., prediction and generation) and demonstrating strong generalization capabilities to unseen cities.

### Strengths
1. This paper proposes a novel location encoder that maps the geographic locations of different cities into a shared semantic space, overcoming the obstacle of previous models that rely on cities and are not scalable across cities.
2. The proposed model employs a spatially aware mixture of expert Transformer architecture to efficiently capture diverse movement patterns driven by different intentions.
3. MoveGPT achieves new levels of SOTA in a wide range of downstream tasks, including prediction, generation, and anomaly detection, and demonstrates strong generalization capabilities.

### Weaknesses
1. The paper claims that its model has global scale and universal capabilities, but its pre-training dataset consists entirely of 16 U.S. cities. The movement patterns of US cities differ significantly from those in Europe or elsewhere, and are likely to be “US models” with no proven ability to generalize to other diverse urban environments around the world.
2. The proposed model maps the entire city to a 500mx500m grid, which is a coarser resolution and will inevitably lose information. Because the same grid area may have different spatial semantic information, but this approach treats it as the same token.
3. This paper claims MoveGPT serves as a foundation model, but only demonstrates that the model can be adapted to a wide range of tasks, and zero-shot or few-shot capabilities are also important for foundation models.
4. The dataset used in this paper does not appear to be publicly available, and the current version lacks a range of details about data sources, construction, and collection.
5.  typos in line 57 and 262.

### Questions
1. This paper normalizes the data for each city, so how is the model able to identify data from different cities?
2. How is the data constructed when pre-training MoveGPT? Is the data randomized across different cities, or does it have a corresponding ratio? How to prevent catastrophic forgetting？

### Soundness
3

### Presentation
3

### Contribution
3

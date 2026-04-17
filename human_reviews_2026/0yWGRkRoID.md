# Massive-STEPS: Massive Semantic Trajectories for Understanding POI Check-ins

- Decision: Reject
- Scores: 4, 6, 2

## Abstract
Understanding human mobility through Point-of-Interest (POI) trajectory modeling is increasingly important for applications such as urban planning, personalized services, and generative agent simulation. However, progress in this field is hindered by two key challenges: the over-reliance on older datasets from 2012-2013 and the lack of reproducible, city-level check-in datasets that reflect diverse global regions. To address these gaps, we present Massive-STEPS (Massive Semantic Trajectories for Understanding POI Check-ins), a large-scale, publicly available benchmark dataset built upon the Semantic Trails dataset and enriched with semantic POI metadata. Massive-STEPS spans 15 geographically and culturally diverse cities and features more recent (2017-2018) and longer-duration (24 months) check-in data than prior datasets. We benchmarked a wide range of POI models on Massive-STEPS using both supervised and zero-shot approaches, and evaluated their performance across multiple urban contexts. By releasing Massive-STEPS, we aim to facilitate reproducible and equitable research in human mobility and POI trajectory modeling. Our code is available at: https://anonymous.4open.science/r/Massive-STEPS/.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper addresses the limitation of existing check-in datasets, which often lack coverage across diverse global regions. To overcome this issue, it introduces a new benchmark dataset called Massive-STEPS. This dataset encompasses data from 15 cities across a wide range of regions, including East, West, and Southeast Asia, North and South America, Australia, the Middle East, and Europe, covering time periods from 2012–2013 to 2017–2018. Moreover, it incorporates semantic attributes—such as POI names, categories, and addresses—that are useful for LLM-based methods. A variety of POI-related models are comprehensively evaluated on Massive-STEPS across multiple urban contexts.

### Strengths
1.	There is currently no widely accepted benchmark dataset, which leads to different models being trained and evaluated on different datasets, making fair comparison difficult. Therefore, developing a new public dataset with recent movement patterns, high diversity, and enriched semantics is highly meaningful for POI-related research.

2.	For POI modeling, existing datasets are often outdated, of low quality, lack regional diversity, or are not publicly available. The paper provides a thorough analysis of these limitations in existing POI datasets.

3.	The paper is clearly written and easy to follow, demonstrating good organization and readability.

### Weaknesses
1.	Although the dataset provides diverse data sources, it remains challenging for models to effectively learn from this diversity in Massive-STEPS. As shown in Table 3, each model is evaluated on a single sub-dataset of Massive-STEPS, resulting in no substantial difference in data usage compared to other datasets.

2.	Massive-STEPS is derived from the STD dataset with several preprocessing steps, such as trajectory grouping, city-level matching, and filtering. However, the transformation from STD to Massive-STEPS appears to be more of a data aggregation process rather than the construction of a fundamentally new dataset. Similar improvements in recency and diversity could potentially be achieved by aggregating existing POI datasets. Thus, the novelty of the dataset is somewhat limited.

3.	The paper would benefit from including additional descriptive statistics—such as the average trajectory length, mean sampling interval, or metrics that capture trajectory sparsity—to provide a clearer understanding of the dataset’s characteristics.

### Questions
1.	The authors claim that existing datasets suffer from low data quality. However, it is unclear how Massive-STEPS addresses or mitigates this issue. More details about the data cleaning or validation process would strengthen the paper.

2.	Does Massive-STEPS include or provide access to a road network map to support map-constrained methods or tasks? Clarifying this aspect would help readers understand the dataset’s applicability to spatially constrained modeling.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
5

### Summary
This paper introduces Massive-STEPS, a large new benchmark dataset for POI trajectory modeling. Its purpose is to address critical limitations in current human mobility research, which heavily relies on outdated datasets from 2012–2013 and is disproportionately focused on a handful of cities like New York and Tokyo. Furthermore, many existing datasets suffer from poor data quality (e.g., GSCD contains nearly 50% erroneous entries) and lack reproducibility. To address these issues, Massive-STEPS is built upon the high-quality STD dataset, providing data across 15 geographically and culturally diverse cities, including understudied regions. The dataset incorporates both recent (2017–2018) and earlier (2012–2013) check-in data, spanning a total duration of 24 months. It is further semantically enriched using Foursquare OS Places data, supplementing metadata such as POI coordinates, names, and addresses. Finally, the authors conduct extensive benchmarking on Massive-STEPS across three tasks, supervised POI recommendation, zero-shot POI recommendation, and spatio-temporal classification, to advance reproducible and equitable mobility research.

### Strengths
1. The first strength of this paper lies in the dataset itself. It addresses a widely recognized, severe bottleneck that has hindered progress in the field. By providing a large-scale, more modern, geographically diverse, and reproducible dataset, this work offers an invaluable service to the community.
2. The author conducted extensive experiments covering a wide range of approaches, from classical models, GNNs, and LLMs. Also, they were tested across three distinct tasks, which provides a highly robust and valuable baseline for future research utilizing this dataset.
3. The dataset provided encompasses 15 distinct cities, particularly including some less popular or previously overlooked regions, marking a significant advancement. Combined with data from 2017-2018, this facilitates more generalized and practically relevant analysis of human mobility.

### Weaknesses
1. Although this paper provides benchmarking and implementation code, consolidating all models used into a standardized code repository would significantly enhance the quality of the work.
2. This paper primarily reviews and integrates existing work, without proposing any independent models or insights.
3. It would be better if the authors compared the performance of the model in terms of changes in the dataset between the 2012-2013 and 2017-2018 time periods.

### Questions
Please refer to the weaknesses

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper presents Massive-STEPS, a large-scale dataset designed to address longstanding limitations in POI trajectory modeling. Specifically, the reliance on outdated, geographically limited, and non-reproducible check-in datasets. Massive-STEPS provides a semantically enriched resource covering 15 cities across diverse global regions and two time periods, enabling both longitudinal and cross-city analyses. The dataset includes rich semantic information such as venue names, addresses, categories, and coordinates. The authors further provide benchmark results for both supervised and zero-shot POI trajectory modeling methods, demonstrating the dataset’s potential utility across different model types and tasks.

### Strengths
1. The paper provides a solid overview of existing check-in datasets and clearly identifies their limitations, offering useful context for the community.

### Weaknesses
1. The contribution appears somewhat incremental. As shown in Table 1, Massive-STEPS seems closely related to Semantic Trails, differing mainly through data reorganization rather than introducing substantial new content or methodology.

2. The authors emphasize that prior datasets are outdated; however, Massive-STEPS itself relies on data from 2017–2018, which remains quite old by 2025 standards and does not fully address the claimed issue of temporal relevance.

3. Although the paper defines three different benchmark tasks, the corresponding model evaluations are limited in both scale and complexity, lacking deeper exploration or meaningful analysis.

4. The writing quality could be improved. While the paper is lengthy, the information density is relatively low—many pages are dominated by large tables or figures containing simple statistics (e.g., Figure 1 visualizes only 14 data points yet occupies a full page).

### Questions
Could the authors elaborate on the weaknesses.

### Soundness
2

### Presentation
1

### Contribution
1

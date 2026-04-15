# Salvador Urban Network Transportation (SUNT): A Landmark Spatiotemporal Dataset for Public Transportation

- Decision: Reject
- Scores: 6, 6, 5, 5

## Abstract
Efficient public transportation management is essential for the development of large urban centers, providing several benefits such as comprehensive coverage of population mobility, improvement of the local economy with the offer of new jobs and the decrease of transport costs, better control of traffic congestion, and significant reduction of environmental impact limiting gas emissions and pollution. Realizing these benefits requires carefully pursuing two essential pathways: (i) deeply understanding the population and transit patterns and (ii) using intelligent approaches to model multiple relations and characteristics efficiently. This work addresses these challenges by providing a novel dataset that includes various public transportation components alongside machine learning models trained to understand and predict different real-world behaviors. Our dataset comprises daily information from about 710,000 passengers in Salvador, one of Brazil's largest cities, and local public transportation data with approximately 2,000 vehicles operating across nearly 400 lines, connecting almost 3,000 stops and stations. As benchmarks, we have fine-tuned diverse Graph Neural Networks to perform inference on vertices and edges, undertaking both regression and classification tasks. These models leverage temporal and spatial features concerning passengers and transportation data. We emphasize the greatest advantage of using our dataset lies in different possibilities of modeling a real-world urban mobility dataset, reproducing our results, overcoming our models, and investigating several other open-problem situations listed in this manuscript as future work, which include the designing of new methods, optimization strategies, and environmental approaches. Our dataset, codes, and models are available at https://github.com/suntdataset/sunt.git.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces SUNT, a comprehensive dataset from public transportation in Salvador, Brazil, with the aim of advancing urban mobility research. It covers detailed data on passengers, vehicles, routes, and geographic features, which makes it a useful resource for various fields like transportation planning, computational modeling, and even managing environmental impacts. By using Graph Neural Networks, the authors show that this dataset can effectively predict transit patterns. It’s great to see that they emphasize reproducibility and potential for future work, which is always crucial for building on research like this.

### Strengths
* It provides detailed data on passengers, vehicles, routes, and geographic features, making it valuable for fields like transportation planning, computational modeling, and environmental management.
* The use of Graph Neural Networks showcases the dataset’s potential in predicting transit patterns.
* The authors emphasize reproducibility and potential for future work, which is crucial for extending research.
* The paper is well-written, with an engaging introduction that sets the context and importance of the dataset effectively.
* The dataset is particularly impressive given the challenges of accessing AVL and AFC data from government sources, making it highly valuable for future research.

### Weaknesses
* Much of the content related to data collection in Section 3 is familiar to the transportation community, lacking novelty for that audience.
* The computer science community, on the other hand, would benefit more from a stronger focus on benchmarking.
* Figure 4 could be more informative by including additional benchmarking tests, as the current single-stop time series is limited in demonstrating the dataset’s predictive capabilities.
* There is a need for more discussion of comparable SOTA benchmarks across different tasks, which would facilitate broader research and model comparisons.

### Questions
* On page 5, line 260, the authors state that “these values were estimated by local specialists based on the passengers’ profiles and the transportation infrastructure in Salvador.” How are passenger profiles integrated into the origin-destination matrix at the regional or stop level?
* Regarding the time range of the dataset, was it specifically chosen for a reason, or is it simply the most recent data available during submission?
* In transportation research, studying mobility changes over longer periods—especially during disruptions like COVID-19—can provide additional insights. Could the dataset be extended to include a longer time range, possibly looking backward? Can the authors justify the current selection?

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
The paper introduces the Salvador Urban Network Transportation (SUNT) dataset, emphasizing its role in enhancing public transportation management in urban areas. It highlights the benefits of effective public transport, including improved mobility, economic development, reduced traffic congestion, and lower environmental impacts. The authors detail the methodologies employed in constructing the dataset, which integrates various spatiotemporal data sources to facilitate comprehensive analyses of public transportation systems.

### Strengths
Originality: The SUNT dataset presents a novel approach to public transportation analysis by integrating spatiotemporal data, which has not been extensively explored in prior studies. This originality lies in both the dataset's construction and its potential applications across various research domains.
Quality: The methodology employed in creating the dataset is rigorous, ensuring that the data is reliable and applicable for future studies. The thorough documentation of processes adds to its overall quality.
Clarity: The paper is well-organized, making it accessible to readers with varying levels of expertise. Key concepts and methodologies are clearly articulated, which aids in comprehension.
Significance: The implications of this work are significant, as the dataset addresses important challenges in urban transportation management. It has the potential to influence policy decisions and improve transportation systems in urban settings.

### Weaknesses
Data Sourcing and Preprocessing: Further details on data sourcing and integration are recommended, specifically covering how each source (e.g., AVL, AFC, GTFS) was merged and standardized, along with measures for data consistency. The handling of missing data and inconsistencies in time or location stamps, particularly during peak traffic times, could be clarified to enhance reproducibility for future researchers.
Experimental Validation and Real-World Applications: The paper could be strengthened by including case studies and examples of real-world scenarios, particularly highlighting the dataset's performance under various conditions, such as peak versus non-peak hours or seasonal variations. This would help substantiate the practical value and adaptability of the dataset in different settings.
Use of the Latest Models: The selection of models, while robust, does not appear to include some recent advancements. The latest models utilized date up to 2022. A few state-of-the-art spatiotemporal and multimodal GNN architectures developed after 2022 were not explored, potentially limiting the demonstration of the dataset's full capabilities. Indicating any constraints or considerations regarding model selection could provide helpful context for future research.
Visualization and Presentation of Results: Additional visual aids could enhance comprehension of complex data trends, such as high-passenger-load areas, traffic flow, or frequent delay points. Developing an interactive dashboard or map to display variations in model performance across geographic regions could also illustrate prediction accuracy or confidence levels in different network zones.

### Questions
Dataset Limitations: Clarifying the specific limitations encountered during the creation of the SUNT dataset, such as constraints in data collection or accuracy, would help understand their impact on the dataset’s applicability in various urban contexts.
Comparative Analysis with Existing Datasets: A comparison with other spatiotemporal datasets could strengthen the paper by positioning the SUNT dataset within the broader field of urban transportation research, highlighting where it advances the state of the art or addresses gaps in existing datasets.
Use of State-of-the-Art Models: A more detailed discussion on how the selected models align with recent advancements in spatiotemporal data modeling, including post-2022 innovations, would enhance the paper’s relevance and demonstrate the dataset’s adaptability to current methodologies.

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
The paper presents the Salvador Urban Network Transportation (SUNT) dataset, which is structured into raw data and a graph that connects various transportation data while respecting geospatial and temporal constraints. It includes four raw datasets collected over five months, detailing information about vehicles, passengers, and stops/stations, as well as a preprocessed dataset for complex network analysis. The authors provide benchmark models using Graph Neural Networks for classification and regression tasks.

### Strengths
+ The paper provides a dataset that includes extensive information about the public transportation system in Salvador, Brazil.
+ The authors provide detailed information on the datasets in the GitHub repo.
+ Very detailed information has been provided in the appendix.This could be useful to beginners in this domain.

### Weaknesses
- Limited to the public transportation system of only Salvador which may limit the overall impact of this work. Providing more detailed case studies or examples of how the proposed dataset can help resolve the current limitations in urban research could illustrate its impact.
- Although a bunch of methods have been included in the benchmark, there are only 2 methods from 2022 and no method from 2023/2024. Considering the fast development in the spatiotemporal AI domain, it would be more convincing to include more recent methods.
- For the first node classification task, all the methods performed pretty well. It feels like a solved problem. Why use this task for the benchmark?
- Including more challenging settings would be interesting, such as anomaly detection, long-term forecasting etc.

### Questions
1. What are the other potential applications of using the 4 raw datasets? In the paper, the 4 raw datasets are used to form an OD dataset. What are the other potentials?
2. In the data collection and dataset formulation process, are there any suggestions/feedback/insights from domain experts such as traffic planners and/or transport operators?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
4

### Summary
This paper introduces SUNT (Salvador Urban Network Transportation), a new large-scale spatiotemporal dataset for public transportation research. The key contributions are:

1. A comprehensive dataset containing information from about 710,000 passengers and 2,000 vehicles operating on nearly 400 lines and 3,000 stops/stations in Salvador, Brazil, collected over 5 months in 2024.

2. The dataset is provided in two forms:
   - Raw data from multiple sources including automatic vehicle location, fare collection, transit feed specifications, and local trip information.
   - A processed graph-based dataset integrating spatial and temporal information.

3. Benchmarking results using various machine learning models, including Graph Neural Networks, for tasks such as node classification, edge classification, and node regression.

4. Pre-trained models and code to reproduce the benchmarking experiments.

5. Detailed methodology for constructing the dataset, including techniques for inferring passenger boarding and alighting locations.

6. Discussion of potential applications and future research directions enabled by this dataset.

The authors emphasize that SUNT allows for modeling real-world urban mobility data in different ways, reproducing their results, improving on the baseline models, and investigating various open problems in transportation research. They provide the full dataset, code, and models to facilitate further research in this area.

### Strengths
Originality:
- The SUNT dataset is a novel and comprehensive public transportation dataset, combining data from multiple sources (AVL, AFC, GTFS, LTI) for a major Brazilian city over a 5-month period.
- The authors develop an innovative methodology to infer passenger alighting locations, which is a common challenge in transportation datasets where only boarding information is directly captured.
- The integration of spatial, temporal, and quantitative data into a complex network representation can enable rich analysis.

Quality:
- The dataset construction process is rigorous and well-documented, with clear explanations of data cleaning steps and assumptions.
- The benchmarking experiments are thorough, testing multiple graph neural network architectures on three distinct tasks (node classification, edge classification, node regression).
- The authors provide pre-trained models and code, enhancing reproducibility and enabling further research.

Clarity:
- The paper is well-structured, with a clear introduction, detailed methodology, and comprehensive results section.
- Figures and tables effectively illustrate the dataset construction process and benchmark results.
- The authors provide extensive supplementary materials, including appendices with additional details on models, metrics, and future research directions.

Significance:
- SUNT addresses a gap in publicly available transportation datasets, providing a large-scale, multi-modal dataset that can drive research in urban mobility and intelligent transportation systems.
- The dataset's comprehensive nature allows for diverse applications, from traffic prediction to route optimization and environmental impact assessment.

### Weaknesses
1. Limited comparison to existing datasets:
The authors mention several existing transportation datasets (e.g., TaxiBJ, BikeNYC, METR-LA) but don't provide a detailed comparison of SUNT's features and capabilities relative to these datasets. A table comparing key attributes (e.g., data types, temporal/spatial resolution, coverage) would help readers understand SUNT's unique contributions and advantages.

2. Lack of error analysis in data processing:
The paper describes a complex process for inferring passenger alighting locations, but doesn't provide an analysis of potential errors or biases introduced by this method. An evaluation of the accuracy of these inferences, perhaps through a small-scale validation study, would strengthen confidence in the dataset's quality.

3. Limited exploration of temporal aspects:
While the dataset includes 5 months of data, the benchmarking experiments focus on only a few specific days. A more comprehensive analysis exploring temporal patterns (e.g., weekday vs. weekend, seasonal variations) would better demonstrate the dataset's potential for longitudinal studies.

4. Narrow range of benchmark tasks:
The benchmarking focuses on three tasks (node classification, edge classification, node regression). Expanding this to include tasks more directly relevant to transportation planning (e.g., demand prediction, route optimization) would better showcase the dataset's practical utility.

5. Lack of multi-modal analysis:
Although the dataset includes information from buses, BRT, and subway systems, the benchmarks don't explicitly demonstrate how these different modes can be analyzed together. Showcasing multi-modal analyses would highlight a key strength of the dataset.

6. Limited discussion of privacy considerations:
While the paper mentions that privacy was respected in data collection, there's little discussion of potential re-identification risks or mitigation strategies employed in the public release of the dataset.

7. Absence of qualitative insights:
The paper focuses heavily on quantitative benchmarks but doesn't provide qualitative insights into transportation patterns or urban dynamics that the dataset reveals. Including some case studies or visualizations of interesting patterns would make the paper more compelling to transportation planners and policymakers.

8. Lack of scalability analysis:
Given the large scale of the dataset, an analysis of computational requirements for processing and analyzing the full dataset would be valuable for researchers planning to use it.

9. Limited discussion of data quality and potential biases:
While the paper describes the data collection process, it doesn't deeply address potential biases or quality issues in the raw data sources (e.g., GPS errors, fare evasion).

### Questions
Same as the weaknesses

### Soundness
2

### Presentation
3

### Contribution
2

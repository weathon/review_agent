# XTraffic: A Dataset Where Traffic Meets Incidents with Explainability and More

- Decision: Withdrawn (Treated as Reject)
- Scores: 3, 5, 3, 5

## Abstract
Long-separated research has been conducted on two highly correlated tracks: traffic and incidents. Traffic track witnesses complicating deep learning models, e.g., to push the prediction a few percent more accurate, and the incident track only studies the incidents alone, e.g., to infer the incident risk. We, for the first time, spatiotemporally aligned the two tracks in a large-scale region (16,972 traffic nodes) over the whole year of 2023: our XTraffic dataset includes traffic, i.e., time-series indexes on traffic flow, lane occupancy, and average vehicle speed, and incidents, whose records are spatiotemporally-aligned with traffic data, with seven different incident classes. Additionally, each node includes detailed physical and policy-level meta-attributes of lanes. Our data can revolutionalize traditional traffic-related tasks towards higher interpretability and practice: instead of traditional prediction or classification tasks, we conduct: (1) post-incident traffic forecasting to quantify the impact of different incidents on traffic indexes; (2) incident classification using traffic indexes to determine the incidents types for precautions measures; (3) global causal analysis among the traffic indexes, meta-attributes, and incidents to give high-level guidance of the interrelations of various factors; (4) local causal analysis within road nodes to examine how different incidents affect the road segments' relations. The dataset is available at https://anonymous.4open.science/r/XTraffic-E069.

## Human Reviews

## Human Reviewer 1

### Rating
3

### Rating Number
3

### Confidence
5

### Summary
The paper introduces XTraffic, a comprehensive dataset that integrates traffic flow data with incident records and road meta-features, covering a large-scale region with 16,972 traffic nodes over the entire year of 2023. The dataset aims to bridge the gap between traffic and incident data, enabling research in traffic forecasting, incident classification, and causal analysis. It includes time-series data on traffic flow, lane occupancy, and average vehicle speed, along with records of seven different incident classes that are spatiotemporally aligned with the traffic data. Each node in the dataset also features detailed physical and policy-level meta-attributes of lanes. The paper presents experiments on post-incident traffic forecasting, incident classification using traffic indexes, global causal analysis among traffic indexes, meta-attributes, and incidents, and local causal analysis within road nodes to examine how different incidents affect road segments' relations. The dataset is available for research purposes and is expected to revolutionize traditional traffic-related tasks towards higher interpretability and practical application.

### Strengths
This paper aims to propose a comprehensive traffic dataset with both traffic and incident data. This dataset is meaningful as there are no similar structured datasets in the literature.

### Weaknesses
1. The dataset does not provide the precise latitude and longitude of incidents. While this information could be inferred from the absolute postmile (Abs PM) and the closest sensor location, not having this data readily available could hinder detailed spatial analysis of incident impacts.
2. The dataset is currently insufficient for cross-year seasonal analysis due to the limited data collected so far. The authors are aware of this and are in the process of collecting and organizing data from additional years to enhance the dataset's capabilities in this area.
3. The dataset's granularity may not be fine enough for analyzing the specific impact of incidents on precise road segments, which could be crucial for certain types of traffic management studies.

### Questions
1. The proposed dataset is based on existing data sources, e.g., incident and traffic data are collected from Caltrans Performance Measurement System (PEMS). The first question arises that since these datasets have existed for a while and some studies have already used both of them for traffic forecasting or causal analysis problems. What is the unique idea of this study compared with these existing studies? The second question is that some previous studies also used some external data sources, e.g., weather and Twitter (now X) data. Why not consider these external data sources too? The third question is that if the proposed dataset is built on existing ones. Then it is out of control for the data quality and the original data problems. For example, "For traffic data, we removed the sensor with less than 50% observations of traffic volume and reserved the data of 16,972 sensors with meta-features." There are many problems in the PEMS data, such as data missing. The final problem is how to determine the data amount is sufficient for the matching between the traffic and incident perspectives or not. If the spatial points are too scare, it is difficult to find both the traffic and incident data in the same spatial and temporal range.
2. The matching method could be an important process for the targeted dataset to be built in this paper. However, the authors fail to give details for their matching method. "We provide two methods for this matching process: (1) involves matching only the nearest sensor on the same freeway as the incident, (2) involves setting a distance threshold and incorporating all sensors within this specified range." Which is the specific method used in the formulated datasets? Or how to decide which method to use? How to set the distance threshold is also not given. It is also difficult to evaluate the matching result since there is no ground truth. The authors fail to verify that their matched results are reasonable. More numerical experiments for the matching methods are expected.
3. The final comments are about the experiments. The authors fail to demonstrate that the proposed combined dataset opens new windows for follow-up studies. The descriptive analysis are very intuitive and the case studies can be conducted with existing datasets. It is difficult to understand how this proposed dataset advances relevant studies.

### Soundness
2

### Presentation
3

### Contribution
2

---

## Human Reviewer 2

### Rating
5

### Rating Number
5

### Confidence
5

### Summary
The paper presents XTraffic, a pioneering dataset that spatiotemporally aligns traffic flow data with incident records across 16,972 traffic nodes over the entire year of 2023. The dataset includes traffic indices (flow, lane occupancy, and average vehicle speed), seven types of incidents, and comprehensive meta-attributes for each traffic node. XTraffic aims to improve the interpretability of traffic-related tasks by integrating traffic and incident data. The authors explore four main applications: post-incident traffic forecasting, incident classification based on traffic indices, global causal analysis of traffic dynamics, and local causal analysis at individual road nodes. The dataset provides new opportunities for research in traffic forecasting, incident detection, and understanding causal relationships in traffic systems.

### Strengths
* The integration of different data sources in XTraffic is a strong point, as it encourages researchers by providing a more holistic view of traffic and incidents.

* The paper compares different tasks related to traffic accidents, presenting a comprehensive analysis of traffic forecasting, incident classification, and causal impacts.

* The benchmarking of different models is well-executed, with references to the work by Liu et al. (2024), ensuring that the dataset is rigorously tested against existing models.

### Weaknesses
* I have mixed feelings about the paper. While the integration of data sources is commendable, I feel the authors haven’t given enough credit to the PeMS system, which seems to be the primary data source. The dataset appears to be more of an extension of PeMS data with additional metadata, rather than a newly collected dataset.

* The work lacks a thorough literature review, particularly of transportation-related research outside of computer science, which weakens the foundation for its claim that integrating traffic data into accident analysis is underexplored. For example, studies by Jin & Noh (2023), Chen et al. (2024), and Liao et al. (2023) integrate both traffic and incident data, challenging the paper’s assertion that this integration is novel.

* The terminology used is confusing. The division between “traffic data” and “incident data” is misleading, as many transportation studies view accidents as part of traffic datasets. The authors’ use of “traffic state datasets” could be clearer, given the broader taxonomy of traffic data that includes both mobility and incident metrics.

* The paper seems limited in scope, focusing narrowly on datasets used in forecasting tasks while neglecting broader applications seen in transportation studies. It lacks insights into the real-world impacts of the dataset for fields beyond computer science.

* The dataset description does not reflect genuine data collection; it mainly involves downloading, processing, and integrating publicly available data, which challenges its classification as a new dataset. The authors’ claim of “collection” is misleading since the data originates from public U.S. city protocols, not direct field observations or deployments.

* The dataset appears suitable only for evaluating the causal impacts of traffic data on accidents, implying a prescriptive approach to how the dataset should be used. This limits potential applications and doesn’t reflect the diversity of research questions found in transportation venues like Accident Analysis and Prevention (AAP).


-----References
* Jin, Zhixiong, and Byeongjoon Noh. "From prediction to prevention: Leveraging deep learning in traffic accident prediction systems." _Electronics_ 12, no. 20 (2023): 4335.
* Chen, Jiaona, Weijun Tao, Zhang Jing, Peng Wang, and Yinli Jin. "Traffic accident duration prediction using multi-mode data and ensemble deep learning." _Heliyon_ 10, no. 4 (2024).
* Liao, Xishun, Guoyuan Wu, Lan Yang, and Matthew J. Barth. "A Real-World Data-Driven approach for estimating environmental impacts of traffic accidents." _Transportation research part D: transport and environment_ 117 (2023): 103664.

### Questions
* Why do the authors claim that integrating traffic and incident data is underexplored, given that such integration is a fundamental requirement in transportation studies?
* How do the authors define the separation between “traffic data” and “incident data,” and why is it necessary for this research? Is this separation justified in the context of broader transportation literature?
* Why is the work presented as a “new dataset,” given that the data mainly come from existing sources like PeMS, and the integration is primarily a matter of processing? Would this work be more suitable as an open-access contribution on platforms like GitHub?
* Are the authors considering expanding the dataset’s scope to be more applicable across diverse research problems, beyond just causal analysis of traffic on accidents? How do they envision its broader use in both computer science and transportation fields?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 3

### Rating
3

### Rating Number
3

### Confidence
3

### Summary
This paper introduces a new transportation dataset that includes both road traffic and incidents by exporting data from Caltrans Performance Measurement System (PeMS) and performing some data cleaning. The primary advantage of this dataset compared to existing ones is that it includes both road traffic and incidents. The paper presents a number of experiments with this paper, starting with a descriptive analysis of the data, forecasting traffic under normal conditions and after incidents, classifying incidents based on traffic, and causal analysis.

### Strengths
* Considering traffic and incidents jointly, and introducing a benchmark dataset is useful.
* Some of the experimental findings are interesting and may lead to future research: the paper shows that traffic prediction performance is significantly reduced after incidents, studies the performance of models for incident classification, and through causal analysis shows what features are important for traffic.

### Weaknesses
* **Data Collection and Cleaning:** The primary contribution of the paper is supposed to be the novel dataset. However, this entire dataset seems to be just an export from Caltrans PeMS. The paper mentions some data cleaning steps, but these are not described in detail, and it is not clear if they are correct or how useful they are (the entire description of the data collection and cleaning is half a page).

* The paper states: "We also collected comprehensive meta-features of these sensors"
Are these also from PeMS?

* "For incident data, we removed repeated incident records"
How were these identified? Why were there repeated records in the data source?

* "For incident data, we removed ... records without absolute postmile"
Why were these removed? What fraction of the data was removed? How complete was the data to begin with (before these removals)?
In general, it is not clear how reliable this dataset is; there is no validation or discussion.

* Was incident duration (used Section 4.1.2) part of the original data?

* **Details for Experiments:** Most of the experiments lack sufficiently detailed descriptions of their methodology. For example, Section 4.2 include zero discussion of references for the methods that are evaluated (they are listed only in the results table). The description of the "Experiment Setting" also lacks detail. 

* Similarly, Section 4.3 lacks information and references on the methods and on the experimental setup.

* **Related Work:** All of the traffic datasets discussed in Section 2 (Related Work) seem to be from a single source, Caltrans PeMS. It would be good if the paper clarified this and introduced PeMS.

* On a related note, the paper includes the following sentence: "Some traffic studies that incorporate incident data use datasets that have not been aggregated or made open source, making it difficult to use them as a standard for evaluating new methods."
There are no references. Which studies does this refer to?

* **Writing:** The quality of writing needs to be improved. While some of the grammatical mistakes and typos are minor, there were multiple parts that were hard to read and understand. For example: "STCL (Liu et al., 2022b) introduced two-month New York City Vehicle incident data as one-hot accident embedding into the prediction of the Taxi and Bike data. Like what we have observed in most works that analyze traffic with incidents (Liu et al., 2022b; Hong et al., 2024), the transport modes of traffic data and that of incident data are NOT seamlessly matched; thus, it will be less convincing to analyze vehicle incidents’ impact on bike traffic (bike lane is separated from vehicle lane) or on taxi traffic (taxi is only a subset mode of the whole vehicle)." (By the way, this dataset does include vehicle traffic and incidents; not just bike traffic.)

* As another example: "XTraffic serves as a rigid testing bed and empirical support to justify model effectiveness and interoperability in deep learning and traffic community."
What does "rigid testing bed" mean? What does "justifying" model effectiveness and interoperability? Do the authors mean evaluating effectiveness and interoperability?

* Early on, the paper states: "We offer a rich collection of physical and policy-level road meta-features."
At this point, it was not clear what "policy-level" meta-features are.

* The description of related work on traffic causal analysis was rather confusing (lines 145 to 158). For example, symbols $X$, $Z$, and $W$ are used without any introduction. The text is very dense.

* What does the "Incident" column of Table 2 represent? This is never made clear.

* It would be more appropriate to call "Physics" meta-features "Physical" meta-features.

* "Since traffic incidents typically affect the traffic on roads, it is viable to deduce the traffic conditions based on the dynamics of the parameters."
What does this mean?

* **Paper Length:** There are some parts that could be omitted shortened. Section 3.1 includes a lot of redundancy, e.g., most of the text between lines 196 and 215 just repeats what has been stated earlier in the subsection. 

* The descriptive analysis of Section 4.1 does not seem particularly useful. Section 4.1.1 just presents some statistics without any discussion. These could be omitted or moved to the appendix.

* Similarly, Section 4.5 does not seem particularly important since it is just an illustrative example.

* **References:** Some references are malformed: "(Department, 2024; of Motor Vehicles, 2023; of Transport, 2016; Huang et al., 2023)"

* Also, the presentation of the references needs to be improved:
"are proposed by (Yu et al., 2018)" should be "are proposed by Yu et al. (2018)", "(Huang et al., 2023) proposes" should be "Huang et al. (2023) propose" (note the conjugation), and so on.

* **Figure and Table References:** Many of these are wrong, pointing to the wrong table, figures, or subfigures. For example, "Table 4" on line 189, "Fig. 2(a)" on line 272, "Fig. 3(b)" on line 274 are all wrong.

### Questions
* What were important data processing steps to create the dataset?

* How were these validated?

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
5

### Rating Number
5

### Confidence
3

### Summary
This paper introduces a new dataset named XTraffic, which provides time-series indexes on Traffic data (traffic flow, lane occupancy, and average vehicle speed), Incident data (Spatiotemporally aligned records of seven different incident types) and Road meta-feature (Detailed physical and policy-level attributes of lanes). The goal of the work is to bridge the gap between traffic and incident data by creating a comprehensive dataset that can improve the interpretability and accuracy of traffic analysis and prediction tasks. The paper demonstrates the effectiveness of XTraffic in four main tasks: post-incident traffic forecasting, incident classification, global causal analysis and local causal analysis. The experiments show that XTraffic significantly improves the performance of traffic forecasting and incident classification models, providing a more detailed understanding of traffic dynamics and incident impacts.

### Strengths
This paper makes an original approach to align the traffic data with the incidents’ records. The experiments part in this paper proves convincingly that this approach could be beneficial for new researches in this field to emerge, showing the significance of this work. The massive analysis and the experiments with recent baseline models greatly validate the work. The work is meaningful for the community.

### Weaknesses
The data are collected by Caltrans Performance Measurement System (PEMS), which leave the main contribution of this paper as the alignment of traffic and incidents. The alignment part has not been sufficiently detailed. It lacks of novelty regarding to ICLR standards.

### Questions
1. There is a typo at the beginning of the section 3.2, for the collection on April 20, 2024, and ended on May 10, 2024. Would it be more like April 20, 2023?

2. As mentioned in the paper, some incidents may affect only one direction of the traffic. However, among the two matching methods to match the nodes and the incidents, the only criteria to consider is the distance, which is not able to tackle the mentioned observation. Could you explicitly explain how the problem could be tackled?

### Soundness
3

### Presentation
2

### Contribution
2

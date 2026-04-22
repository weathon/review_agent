# UrbanGraph: Physics-Informed Spatio-Temporal Dynamic Heterogeneous Graphs for Urban Microclimate Prediction

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 6, 6, 4

## Abstract
With rapid urbanization, predicting urban microclimates has become critical, as it affects building energy demand and public health risks. However, existing generative and homogeneous graph approaches fall short in capturing physical consistency, spatial dependencies, and temporal variability. \revise{To address this, we introduce UrbanGraph, a framework founded on a novel structure-based inductive bias. Unlike implicit graph learning, UrbanGraph transforms physical first principles into a dynamic causal topology, explicitly encoding time-varying causalities (e.g., shading and convection) directly into the graph structure to ensure physical consistency and data efficiency. Results show that UrbanGraph achieves state-of-the-art performance across all baselines. Specifically, the use of explicit causal pruning significantly reduces the model's floating-point operations (FLOPs) by 73.8\% and increases training speed by 21\% compared to implicit graphs. Our contribution includes the first high-resolution benchmark for spatio-temporal microclimate modeling, and a generalizable explicit topological encoding paradigm applicable to urban spatio-temporal dynamics governed by known physical equations.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The authors propose a novel method to generate high resolution forecast of weather variables over cities. The method employs a physics informed graph representation that explicitly maps physical processes.
The authors propose to discretize GIS features and then explicitly express the relationship between cells in five different edge types: Vegetation Evapotranspiration, Shading, Convective Diffusion, Semantic Similarity, Internal Contiguity.
These different edge types govern how the different cells are connected to each-other.

The proposed method is trained on and compared to a high resolution numerical model output (ENVI-met), and performs better than several other baseline methods.

### Strengths
The paper is well written, the advantages of the methods clear and the figures of very high quality. The arguments for the design choices are sound, and make sense in the context of the paper.
The five types of edges seem to accurately represent the different processes, and suffice to express the complex relationships between nodes.
The model does seem to perform better than a series of baselines both in accuracy and time cost.

### Weaknesses
Although the paper is quite strong, I feel like a few key weaknesses exist:
- The authors claim that previous methods mostly use fixed graph structures for time series forecasting. Although this might be true, I would argue that their method is also fixed. The graph is a gridded mesh, where the edges weight are varying. Additionally, the advantage of using GNNs is usually to work with non-gridded data. Since the domain is discretized, what is the advantage of using a GNN? Wouldn't a transformer with a weighted attention mask work as well?
- There is no mention of spatial resolution. Maybe I've missed it, but what is the resolution? The GIS data is discretized, so there is a spatial resolution.
- I am not sure I understand what the inputs are. Beyond the discretized GIS data, what are the inputs of the model? How are the environmental variables integrated? I also didn't see any mention of embedding the environmental variables. The only mention is the Graph-level global environmental features. Are the environmental features only embedded at the graph level? Is there no per-pixel data?
- I have only seen a train a validation set mentioned, but not test set.
- The authors claim to increase the computational efficiency in FLOPS by 17%. But in table 1, GGAN-LSTM performs better in terms of FLOPS. Where does this come from?
- In table 1, URBANGRAPH is bold for most results, but several models beat it for time cost.


Minor weaknesses:
- Figure 4:
	- a) it is very uncommon to show train and validation results. Results are usually shown on the test set.
	- a) What is the x axis?
	- b) the colors are hard to differenciate
	- b) where is URBANGRAPH?
- Figure 3: I don't understand what the discretized images are (top left)
- Line 256: "normalized static feature space." which normalized static feature space?
- Line 379: MAE is not reported

### Questions
I am not sure I understand how the final network is constructed, can i get an example of network?
I would like to see examples of results in the main paper.

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
This paper presents UrbanGraph, a physics-informed framework for urban microclimate prediction using dynamic heterogeneous graphs. The authors propose encoding multiple physical processes (shading, vegetation evapotranspiration, convective diffusion) as different edge types in a heterogeneous graph structure, where edges are dynamically reconstructed at each timestep based on environmental conditions. The framework combines Relational Graph Convolutional Networks (RGCN) for spatial feature extraction with LSTM for temporal modeling. The authors evaluate their approach on UMC4/12, a new dataset containing 11.9 million high-resolution spatio-temporal data points generated from ENVI-met simulations across 11 urban sites in Singapore. The results show improvements of up to 10.8% in R² and 17.0% reduction in FLOPs compared to baselines.

### Strengths
1、Well-motivated physics-informed design:
The explicit encoding of physical processes into graph topology is intuitive and grounded in urban climate science. The distinction between static edges (semantic similarity, internal contiguity) and dynamic edges (shading, vegetation activity, convective diffusion) appropriately captures both time-invariant spatial relationships and temporally evolving physical phenomena.
2、Dataset contribution: 
The UMC4/12 dataset with diverse urban morphologies and high-resolution simulations provides a valuable resource for the research community.
3、Clear presentation:
The paper is generally well-written with good use of figures to illustrate the framework and results.

### Weaknesses
the explicit encoding of physical processes (such as shading, vegetation evapotranspiration, and convective diffusion) into a dynamic, heterogeneous graph topology is a novel and effective method. however, this heavy reliance on predefined rules raises critical questions about the model's flexibility and potential for scientific discovery.  The model is constrained by the five predefined relationship types. As the authors correctly acknowledge in the limitations section, this may oversimplify the real, complex physical interactions.

### Questions
1 The ablation studies and baseline comparisons effectively demonstrate the superiority of UrbanGraph over simpler models like standard GCNs (proving the value of heterogeneity) and non-graph models like CGANs. However, the paper is missing a comparison against more advanced, integrated spatio-temporal Graph Neural Networks (e.g., STGCN, Graph WaveNet, ASTGCN, or ST-GNNs designed for traffic). These models often feature more sophisticated spatio-temporal fusion mechanisms than the sequential RGCN-then-LSTM approach used here. Could the authors comment on why these state-of-the-art spatio-temporal GNNs were not included as baselines? A discussion (or ideally, a new experimental comparison) would be needed to truly position UrbanGraph's performance within the broader ST-GNN literature, rather than just against its own ablated components.
2 Some parameters are likely effective for the UMC4/12 dataset, which is based on Singapore's climate. a) How sensitive is the model's accuracy to these specific values? b) If this model were deployed in a city with a vastly different climate and morphology (e.g., a dry, arid climate with sparse vegetation or different wind patterns), would these parameters need to be manually re-tuned? c) Does the model risk failing if the true physical influence (e.g., of wind) differs from the hard-coded heuristic?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 3

### Rating
6

### Rating Number
6

### Confidence
3

### Summary
This paper proposes UrbanGraph, a framework for urban microclimate prediction that integrates physics-informed, dynamic, and heterogeneous graph neural networks. The key idea is to explicitly encode time-varying physical processes, such as shading, vegetation evapotranspiration, and convective diffusion, into the topology of a dynamic heterogeneous graph. This graph structure is then processed by a spatio-temporal model combining a Relational Graph Convolutional Network (RGCN) for spatial dependencies and an LSTM for temporal evolution. Moreover, the authors also curate the UMC4/12 dataset, a high-resolution, physics-based simulation benchmark. Empirical experiments demonstrate that UrbanGraph outperforms various strong baselines.

### Strengths
1. The problem is well-motivated.
2. The proposed idea of encoding observational/structural bias directly into the topology, i.e., physics-informed edges rather than physics-informed losses, is principled and technically sound.
3. The dataset contribution is valuable to the community.
4. The paper is well written.

### Weaknesses
1. The method is evaluated on a CFD-style simulator (ENVI-met) under several urban configurations. This is fine for an anonymized submission, but the key claim is physics-informed generalization to realistic urban microclimates. Without any real-world/field-sensor validation, it’s hard to tell if the hand-crafted dynamic edge rules are robust to noisy or incomplete inputs. Would it be possible to deploy on real city data (even small-scale)?
2. The model uses an LSTM as the temporal evolution module. They compare to GRU and Transformer, but the argument for LSTM as the final choice is mostly empirical. Given that the graph itself is dynamic, a temporal attention or cross-time graph operation might better exploit the structured changes in edges.

### Questions
1. How expensive is rebuilding 3 physics-driven edge sets every hour for a large city-level grid (say 50k–100k nodes)? Or have you considered or tested any strategies to manage this complexity?
2. You report on six target variables (UTCI, PET, AT, MRT, WS, RH). Are they trained with a single shared UrbanGraph and separate heads, or trained separately per target (Eq. (1) suggests separate mappings)? If separate, could multitask training further boost R² through shared spatial embeddings?

### Soundness
3

### Presentation
4

### Contribution
3

---

## Human Reviewer 4

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
The paper introduces UrbanGraph, a physics-informed framework to predict urban microclimate. UrbanGraph integrates five physical representations including veg. evapotranspiration, shading, diffusion, similarity and internal continuity and encodes them using a relational GCN. Then, the spatial and temporal contexts are modeled using MLPs and LSTM, along with the physical constraints to predict future urban microclimate. Experiments are conducted using the simulation dataset from ENVI-met, and seven baselines are used to demonstrate the improved effectiveness and efficiency of the proposed model.

### Strengths
S1: High-resolution urban simulations and modeling is an interesting and timely topic for digital twins.

S2: Examples are well prepared and help present the main idea and workflow.

S3: Experiments are conducted using 7 different methods showing improvements.

### Weaknesses
W1: The physics-informed contribution uses different physical representations like shading. This is reasonable but the methodological novelty is not well explained. It seems more to be an application that uses basic physics knowledge to define the graph and the physics is specific to this problem.

W2. The experiments feel like an ablation study. More standalone and recent methods should be included for comparison. It uses only a single dataset or comparison. Using additional datasets can help evaluate the generalizability of the model.

W3: In the appendix, the tree and building layers are the same for the second row in Figure A1. This raises concerns about data correctness.

### Questions
It will be helpful if the authors can clarify what the technical contributions are for the edge designs in Figure 1.

### Soundness
3

### Presentation
3

### Contribution
2

# Understanding the Geospatial Reasoning Capabilities of LLMs: A Trajectory Recovery Perspective

- Avg Score: 2.50
- Decision: Withdrawn (Treated as Reject)
- Scores: 4, 2, 2, 2

## Abstract
We explore the geospatial reasoning capabilities of Large Language Models (LLMs), specifically, whether LLMs can read road network maps and perform navigation. We frame trajectory recovery as a proxy task, which requires models to reconstruct masked GPS traces, and introduce GLOBALTRACE, a dataset with over 4,000 real-world trajectories across diverse regions and transportation modes. Using road network as context, our prompting framework enables LLMs to generate valid paths without accessing any external navigation tools. Experiments show that LLMs outperform off-the-shelf baselines and specialized trajectory recovery models, with strong zero-shot generalization. Fine-grained analysis shows that LLMs have strong comprehension of the road network and coordinate systems, but also pose  systematic biases with respect to regions and transportation modes. Finally, we demonstrate how LLMs can enhance navigation experiences by reasoning over maps in flexible ways to incorporate user preferences.

## Human Reviews

## Human Reviewer 1

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
This paper investigates whether LLM can perform geospatial reasoning by reconstructing masked segments of GPS trajectories from road network context. The paper proposes a two-stage prompting framework, path selection followed by coordinate generation. Experiments show that LLMs outperform some baselines, generalize zero-shot across regions, and demonstrate structured reasoning over road networks. Further analyses reveal regional and activity-type biases and show that LLMs can integrate user preferences into route planning.

### Strengths
The introduced GLOBALTRACE benchmark is diverse and realistic.

The evaluation metrics go beyond standard MAE, providing more interpretable path-level measures such as PoT and PoTF1.

The paper is well-structured.

### Weaknesses
The method focuses on prompting design, but does not report any inference or token cost. Since some configurations involve 10K~20K tokens per query.

In Stage 1, the model receives a pre-selected local road network, which already restricts the possible routes to a small area, which may reduce task difficulty and partly explain its high accuracy compared to TrajFM.

Stage 2 relies on explicit road geometries, which introduces a strong spatial prior and reduces the need for actual geographic reasoning.

The current small/large gap split provides only a coarse difficulty granularity: the large-gap range is broad, and longer masked segments would more effectively evaluate genuine geospatial reasoning. In addition, the paper does not report how many samples fall under each difficulty level, which would be important for assessing the balance and relative contribution of each subset to the overall results.

### Questions
How much time does the two-stage prompting framework take to recover one trajectory in practice?

Does this strong geometric prior in both stages simplify the task (geospatial reasoning) too much?

In Fig. 2a, the average “Full trajectory (km)” differs between the small- and large-gap subsets, although each trajectory is said to have both masking variants. Could authors clarify these?

### Soundness
1

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
4

### Summary
The authors attempt to show that LLMs can reconstruct trajectories to the same degree as traditional mapping platforms. This paper does too much though and the authors need to break the paper into different focus area.

### Strengths
Using trajectories to test whether LLMs can spatially reason is a good idea.

### Weaknesses
*The prompts were extremely detailed and there were no prompt sensitivity analyses. The authors want to claim that they showed the LLMs were on par with Google Maps but the inputs were very different and the detailed prompts all but ensured the systems would perfir reasonably.

* The research questions were introduced in the results section, and were never motivated. 

*Not enough detail was provided on exactly what trajectories needed to be generated. Appendix B was too light on details.

* Results of different LLMs fell in the range of 50-60% in the best cases, this is barely better than flipping a coin so these results do no engender confidence.

*The authors claim to show their approach can incorporate user preferences, but just using a couple of examples is not sufficient, especially for their claims. And the authors fail to consider that humans do not consider trajectory reconstruction.The authors should focus on this study separately in another paper.

It was not clear why cycling, bus, and driving was included in the same study as pedestrian activities, such analyses necessarily require different map resolutions and none of this information was included. For the authors to make the claims they did, they need to include all information so people could replicate their studies.

### Questions
*Why would people want to use LLMs for path planning when deterministic algorithms do this quite well?

*Does the level of effort required in setting up the prompts to generate outputs that LLMs can handle justify the effort, especially in light of the success of traditional trajectory planning approaches?

*How brittle is this approach in light of slight prompt variations?

### Soundness
1

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
The paper explores to understand the geospatial reasoning capabilities of large language models. They propose a benchmark named GlobalTrace consisting of 4000 real-world trajectories, LLMs are tasked with reproducing masked segments of a trajectory i.e a sequence of longitude and latitude points using a two-stage task description. The evaluation shows that existing llms exhibit a high degree of geospatial awareness similar to existing algorithms such as those used by Google Maps.

The key contributions of this work is the design and release of a new benchmark to assess geospatial reasoning capabilities.

### Strengths
- The paper proposes a new benchmark to assess geospatial reasoning capabilities of llms
- The benchmark is created with diverse range of trajectories from real world coordinates, and tries to reduce data imbalance and biasness.
- This can be a good addition to existing llm benchmark suites and provide a unified evaluation framework.
- The paper is clearly written and explains in detail the steps used for creating the benchmark and evaluating models.

### Weaknesses
- The authors claim that there have not been previous works that evaluate the ability of llms to plan paths or read road networks or generate coordinates. This is false as there are multiple existing works in this domain such as (https://arxiv.org/pdf/2306.00020) who have already shown that llms contains a strong degree of groundness to real world data and can reason and plan using geospatial data. 
- The benchmark only evaluates the capability of llms in using their internalized knowledge and world coordinates. I am not sure how this might be useful in real world applications, as one can easily call an external api or service to convert between locations and coordinates. Moreover llms are still not reliable in generating very accurate and fine floating point values such as longitude and latitude coordinates, as exhibited by the high MAE scores. The benchmark can be more useful if it addresses how language agents with access to external tools might perform in this scenario.
- Being able to recover a trajectory does not guarantee a high degree of understanding or reasoning capability. It shows that the llm was trained with a lot of geospatial data and can retrieve that knowledge given the correct prompt. Tasks such as RQ4 are more relevant to understanding whether the llm can actually use the knowledge and apply it to solve other tasks such as personalized recommendation. I think the authors can improve the benchmark much more by designing tasks such as these that require multi step reasoning, rather just recalling geospatial coordinates from memory.

### Questions
The paper addresses most of the relevant questions.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper explores whether LLMs can perform geospatial reasoning by reconstructing masked GPS trajectories from road network data. It introduces GLOBALTRACE, a diverse dataset of 4K real-world trajectories, and proposes a two-stage prompting framework for path selection and coordinate generation. Experiments show that strong LLMs outperform specialized trajectory models and approach Google Maps’ performance, revealing both reasoning ability and regional biases. The paper also showcases preference-aware navigation scenarios, highlighting new applications of LLM-based spatial reasoning.

### Strengths
- Introduce a new task for geospatial reasoning via trajectory recovery
- Constructs a comprehensive dataset (GLOBALTRACE) across regions and transportation modes
- Design a prompts to tackle
- Comp[are with LLM and non LLM HMM, Google Maps, TrajFM, etc.)
- Clear metric design (MAE, PoT, F1 variants) with detailed ablations

### Weaknesses
- This work lacks the clear motivation and its impact, other than can LLM do this? And how is it distinct from path finding or routing or navigation given source and destination in term of consequence? 
- No visual or multimodal grounding—limited to text-based reasoning. While the application seems valuable, the current text form is not much, rather I think a multi-modal task would be more significant.  
- The benchmark is seems easy, not challenging, on par the Google maps-which limits its applicability. No clear motivation why to use LLM instead of these methods. Also High token/inference cost limits real-world scalability but there was no study analysis.
- No discussion on human performance as well. 
- Section seems not method rather evaluation protocol. 
- Data collection section with details is missing  
- For Preference aware path finding analyis, Popular path finding is a common task, why not sudy on that?
- It does not discuss why or how the two stage prompt is derived, neigher consider any other reasoning prompts. 
- I dont understand why the dataset is based on OpenStreetMap and the evaluation consider Google maps. 
- Bias discussion shallow—not linked to data imbalance quantitatively
- Missing robustness tests on noisy or incomplete maps or any such discussion
- Not much intuition on error corrections, why it was making the error and how to fix.

### Questions
See weakness

### Soundness
2

### Presentation
2

### Contribution
2

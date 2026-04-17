# SPhyR: Spatial-Physical Reasoning Benchmark on Material Distribution

- Decision: Reject
- Scores: 2, 8, 2, 2

## Abstract
We introduce a novel dataset designed to benchmark the physical and spatial reasoning capabilities of Large Language Models (LLM) based on topology optimization, a method for computing optimal material distributions within a design space under prescribed loads and supports. In this dataset, LLMs are provided with conditions such as 2D boundary, applied forces and supports, and must reason about the resulting optimal material distribution. The dataset includes a variety of tasks, ranging from filling in masked regions within partial structures to predicting complete material distributions. Solving these tasks requires understanding the flow of forces and the required material distribution under given constraints, without access to simulation tools or explicit physical models, challenging models to reason about structural stability and spatial organization. Our dataset targets the evaluation of spatial and physical reasoning abilities in 2D settings, offering a complementary perspective to traditional language and logic benchmarks.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
In this work, the authors explored LLM's reasoning ability on topology optimization problems as a new benchmark. The problem is defined as material distribution in grid cells, under the constraints of fixed points and applied load. Through some 2D test examples, they concluded that LLMs so far have limited understanding of spatial physical reasoning, and advocated more future research to be done in this area.

### Strengths
- The problem definition and setup is very clear.
- The material is easy to follow.

### Weaknesses
Overall, I think the contribution of this work is very limited. It does not invent new techniques, or improve any existing frameworks. So I consider it well below the par of acceptance. Here to list a few weaknesses:

- The quantitive analysis is not very convincing. Since the solution is in form of binary values, 0s and 1s, there is 50% chance LLM can have a right guess. To me it is hard to tell if a model actually has a good physical-spatial understanding or it relies on random guess, especially for those with 50% matching score.
- Some topology optimization problems can have multiple plausible solutions. The evaluation presented does not consider the cases when a model outputs an alternative solution.

### Questions
- What is your definition of difficulty level based on? Size of the domain or number of random cells, or boundary conditions?
- Is it possible to test on 3D problems, and how about domains of other shapes?
- Can you explain why some models perform better than the others on this task?

### Soundness
2

### Presentation
2

### Contribution
1

---

## Human Reviewer 2

### Rating
8

### Rating Number
8

### Confidence
2

### Summary
This paper introduces SPhyR, a benchmark dataset that evaluates Large Language Models' spatial and physical reasoning capabilities through topology optimization tasks. Models must predict optimal material distributions in 2D grids given boundary conditions (loads and supports) without access to simulation tools, with tasks ranging from filling single masked cells to predicting complete structures.

### Strengths
- Addresses an underexplored gap in LLM evaluation by testing physically-grounded spatial reasoning rather than just linguistic or visual tasks.
- Provides graduated difficulty levels (easy/hard) and multiple task variants (cell, row, column, full structure), enabling granular analysis of model capabilities
- Evaluates 10 state-of-the-art models with multiple experimental setups (rotations, prompt variations) and well-defined metrics

### Weaknesses
- Restricted to small 10×10 grids and relatively simple 2D scenarios, which may not fully reveal model limitations or generalize to realistic structural problems.
- Topology optimization can have multiple valid solutions; the paper doesn't address whether alternative plausible structures should be considered correct

### Questions
See Weaknesses.

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
3

### Summary
The authors introduce SPhyR a new benchmark for evaluating spatial and physical reasoning ability of  LLM about material distribution. LLMs are provided with 2D boundary conditions, applied loads and supports, and must predict the stable material layout that satisfies those constraints.

### Strengths
The main strength of the paper is that: 

- It is the first benchmark for this particular task of reasoning about optimal material distribution.

### Weaknesses
The main weaknesses in the paper are as follows:
- It is unclear why would we need to benchmark LLMs for this particular task ? Are LLMs expected to provide more optimal answers than numerical solvers ? Is it going to be faster? 
- If we need to empower LLMs with physical reasoning ability, doesn't it make more sense to just give LLMs access to a numerical solver tool or a simulation tool ? Why is there a need for providing LLMs with the intrinsic ability to solve a physics problem ?
- There is also no clear motivation as to why LLMs could be better than regular ML methods that rely on PINNs or CNNs. 
- Figure 4 and Figure 5 are poorly presented with very small fonts.

### Questions
Why is this task important for LLMs specifically ? What will LLMs offer better than traditional solvers or PINNs or CNNs ? I understand that this a benchmark paper, but the authors should clarify what scientific insight or practical capability we gain if future LLMs achieve high performance on SPhyR.

### Soundness
2

### Presentation
2

### Contribution
2

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
This paper provides a spatio-physical benchmark for LLMs, derived from 2D topology optimization problems with a variety of load and fixed support configurations. The dataset includes a ground truth material distribution (as computed by a commercial software suite) and corresponding queries. Each query provides a partially- or fully-masked version of the ground truth, together with a natural language prompt requesting that the model complete the missing elements in a way that uses minimal material while supporting the specified load. The authors benchmark several models using this dataset to form a preliminary baseline, and provide some discussion of the models' performance and common pitfalls. The paper also includes some additional experiments testing alternative formulations, such as rotated examples or query formulations featuring more or less physical intuition.

### Strengths
The premise of the paper is intriguing and useful as a general benchmark, as it would certainly be beneficial to query whether models can reliably reason over physical forces, constraints, material interactions, and structural connectivity. Topology optimization seems like a great task through which to measure such reasoning capabilities for VLM/LLMs, especially because the pixel/voxelized format of the problem naturally translates to several modalities, including text serialization as posited in this paper.

The gradation of tasks is clever and poised to provide nuanced insight when used as a benchmark. There are also ample opportunities for extensions based off of this dataset, such as using the samples for curriculum learning to impart progressively more structural knowledge in a model. 

The paper is positioned well within existing literature, including a very informative overview of related approaches and their distinctions. 
The paper is well structured and illustrated, including enough detail for a non-domain expert to follow it.

### Weaknesses
1. My main concern is that the evaluation metrics seem far too simplistic.
	- This is especially true in the "Score" metric for the harder continuous scenario, where, for a ground-truth cell with value 0.6, model predictions of 0 or 0.7 would be marked equally wrong. This sort of all-or-nothing scoring approach based on string matching seems to encourage pattern matching or memorization over physically-based reasoning -- which is precisely the issue this paper claims to address. Relative differences would be a good start. But more critically, it seems like at least one evaluation metric should incorporate a physical simulation -- or at the very least, a force path analysis -- to assign a score based on functionality. 
	- It also seems odd to equally penalize errors in cells that were originally masked vs. those that were fixed but the model changed anyway. At worst, fixed-but-changed deviations seem like flagrant disregard for (or forgetting of) the task, so they should be penalized harshly. At best, I'm curious: are there any cases where it might change them to arrive at a different but equally valid or better solution? In that case, they indicate great physical understanding and global reasoning, so perhaps they should be rewarded according to the physical performance (though perhaps still with some penalty for rule-breaking).
	- The discussion in Section 6.2 seems like it comes closer to the core of the task evaluation than any of the official metrics. For example, building the isolated material islands clearly suggests a lack of physical reasoning; a metric that identifies and penalizes this sort of behavior seems far better poised to tease out and rank/reward the model's underlying abilities. Consider also the following cases: (1) a masked cell that mimics all of its surrounding neighbors, where an easy pattern recognition like flood fill would yield the right answer; (2) a masked cell that opposes all of its neighbors, where something like flood fill would be wrong; and (3) a masked cell on an established boundary, where the neighbors are split and it's unclear which way to go. A model's correct prediction on cases (2) or (3) should be more highly rewarded than case (1). If the paper were to develop metrics based around such domain-specific observations, this dataset could offer considerably more than it does in its current form.

2. The formulation of topology optimization used in this dataset is somewhat unclear. Could you clarify the the precise objective that you used in Rhino, and comment on whether/how well that aligns with the one sought by your prompts? Often, topology optimization is "minimum compliance subject to a maximal volume", which usually leads to a well defined solution. However, the phrasing on l.135 suggests that compliance and volume should both be minimized -- leading to multi-objective optimization with a Pareto front of possible solutions. On l.149 (and in the primary prompt template), the phrasing seems to imply something different still: the minimal volume that provides a structure capable of withstanding the forces (with an unspecified relationship between the acceptable level of compliance/stiffness that constitutes "withstanding"). Only the Physics-Enhanced prompt in Appendix F.2 seems specific: as little material as possible, with at least one complete load path. I imagine that each one of these formulations might lead to slightly different TO solutions (and thus, confusion within the benchmark if there are multiple, conflicting, or unclear answers).

### Questions
1. I wonder how a more explicitly visual medium might impact performance on this task -- e.g., I imagine the tokenization/attention is different for a text-based grid vs. an explicit 2D image domain. Have you considered a parallel benchmark that supplies images in a VLM? If you wanted to stay in a text based domain, I'd be curious whether/how the LLM response differs if the grid is framed as e.g. a numpy array. I also suspect that the row/column bias discussed in Section 6.1 might be due to the serialized text grid, as rows remain together but columns require models to attend to information that is likely spread over more distant tokens. 

2. What motivated your choice to have the model return a full material array, rather than simply returning the masked elements. Did you try the latter, and if so, were there any notable differences?

3. Does the Grasshopper/Millipede solution always converge to a suitably correct answer within the allotted iterations? Do you check for convergence before adding to the dataset? Is there always 1 unique solution for a given setup, or can there be multiple?

4. How many distinct TO configurations are contained in the dataset? It would also be interesting to have a breakdown regarding e.g. how many have holes in the structure (to give a sense of solution complexity, and whether simple pattern recognition like floodfill might often work to solve topology optimziation without necessarily needing physical reasoning)

5. Could you provide additional information about the 100 examples used for evaluation? e.g., how many of each task type? how many distinct TO configurations are represented? 

6. The interpretation of experiments is often a bit sparse. For example, there is no analysis of the results shown in Figure 5, aside from a brief mention in the discussion.


## Minor comments 
- l. 55, 86 -- This phrasing strikes me as a specialized argument that undersells your intention, and is somewhere in the ballpark of a tautology... "these benchmarks don't test performance on topology optimization, because they don't ask about topology optimization, but we do". It might be worth revisiting, to see if you can formulate a stronger motivating point.
- l. 210,215 -- the explicit mention of binary values was confusing, since you also have continuous-valued versions.
- l. 453 - this is the first and only mention of 3D samples in the paper to this point, which was jarring.  
- Figure 2 could be moved to the supplement 
- l. 287 - should be Figure 3?
- l. 303,304 - correction --> completion
- Fig 7 (Appendix) -- could you explain the meaning of the differently colored dots? 
- I wonder if the terms "void" and "support" are misleading; I consistently find myself interpreting them as "no material" and "material" even though I know that's incorrect. Altering them in the prompts might have an effect on the model responses

### Soundness
2

### Presentation
3

### Contribution
2

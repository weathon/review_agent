## Human Reviewer 1

### Summary
This paper proposes a benchmark that pairs real measurements with matched numerical simulations across five scenarios (cylinder wake, controlled cylinder, FSI, foil, and combustion; governing equations span Navier-Stokes, coupled FSI, and reactive Navier-Stokes with species transport). The benchmark defines three training regimes (train on simulation, train on real, pretrain on simulation then finetune on real), includes a mix of pixel and physics flavored metrics, and evaluates a set of neural PDE baselines including a pretrained foundation model. The headline empirical messages are: there is a nontrivial gap between simulation and laboratory data; pretraining on simulation generally helps downstream on real; and the codebase makes it straightforward to add models or datasets.

I think this is timely and potentially useful. If we are serious about sim2real for scientific ML, we need carefully curated real data and a shared protocol. However, the current paper mixes benchmarking with a sim2real narrative in ways that are a bit loose, the experimental details are too thin for others to trust or extend the datasets, and some of the metrics and literature framing are not well aligned with fluid mechanics and combustion practice.

Am open to potentially increasing my score if the authors narrow the claims, remove the robotics sim2real framing argument and instead point to real sim2real problems in fluid dynamics, add significantly more context to place this in the existing fluid dynamics literature, and substantially improve the experimental documentation and physics grounded evaluation.

### Strengths
- The benchmark collects paired real and simulated trajectories for several nontrivial systems instead of yet another synthetic-only PDE suite. This is likely to be useful for the community.
- The split of training regimes (simulation only, real only, pretrain on simulation then finetune on real) is useful, and the pretraining result is consistent with what many of us have seen in practice.
- The code appears modular enough to add a new dataset or baseline without painful surgery, and using a single file format lowers friction for adoption.
- Including both data oriented metrics and physics oriented diagnostics is better than reporting only RMSE. The autoregressive evaluation option is also a good idea.
- The combustion scenario is ambitious and, if documented properly, could become a valuable stress test beyond the usual laminar toy problems.
- The baseline measurements for their benchmark are very extensive.

### Weaknesses
Major concerns:

- The documentation about the experiment is unacceptably thin in its current form. If the experimental data was created or modified from another source, you need to cite it. If someone else created the dataset for you, you need them to write documentation for it. The current documentation on experimental data generation (which, of course, can be included in the appendix) is simply unacceptable for publication, especially for a paper which is supposed to be about this very dataset.
- The paper motivates the sim2real gap by citing mostly work from robotics, which I found very strange, almost as if the authors are guessing there is a sim2real gap in fluids, without actually surveying the literature. Robotics has a much different sim2real gap than turbulence research does. In fluids, the sources of discrepancy, data acquisition, and noise models can be quite different. Please ground the narrative in fluids and combustion references. If you are addressing the sim2real gap in fluids, you must speak from the context of the fluids community, and discuss the ways in which the fluids community has quantified this gap.
- Please state clearly whether you will release raw data (e.g., the PIV frames), calibration files, and the full processing scripts, not just the final HDF5 arrays, so that it can be checked by others. If raw data cannot be released, say so and justify it. Benchmarks live or die by their data hygiene, and biases in a benchmark can leak into biases in the community's preferred models.
- Several figures quantify differences using frequency or Fourier errors over image-like arrays. This is admittedly a start, but it is not a physics grounded measure of mismatch between experiment and simulation, and certainly not something people in the fluids community would actually use as a robust measure of discrepancies between simulation and real data. In fact, for several real-world experimental problems, there is a difference here that is simply due to the nature of the real-world experiment, yet the simulation and experiment can actually have no discrepancy. This is because you would only care about some summary statistic, and not care about some wave mode that you know does not affect your statistic. Yet, your metric would completely miss this. I recommend reviewing and citing the fluids literature and how people measure discrepancies between simulations and real data. Note that this is a problem people have studied for literally decades in the fluids community. These additions would make your claims about a "gap" more convincing.
- Simulation contains modalities that are not observed in the lab, and the current strategy randomly masks channels and adds noise. That is a start, but it does not reflect the actual sensor physics. Please consider sensor specific degradations (camera noise models, optical blur, saturation, PIV algorithmic artifacts) and state explicitly which channels are used for training and which are hidden. It would also help to define tasks that force parity, for example training all models only on the modalities that the lab provides.
- The paper claims to be the first benchmark that integrates real-world measurements with paired numerical simulations across complex physical systems. Within fluids, this is far from true; as just one demonstration, "ERCOFTAC" has hosted combined experimental and numerical reference cases since 1995 across a wide range of flows. Please narrow the novelty claim to something that accurately represents existing datasets and benchmarks, and perhaps cite existing databases.


Additional comments and suggestions

- The update ratio metric is interesting, but it conflates pretraining data scale with optimization effects.
- Please make the train, validation, and test splits explicit at the parameter level so that generalization across Reynolds number, control frequency, mass ratio, or equivalence ratio is clear. I consider those to be most interesting axes. And if you already do this, show it more prominently.
- The autoregressive evaluation stops very early. If you want to make claims about stability, show longer horizons and add probe based diagnostics, not only field RMSE.
- The baselines are modern ML models, which is fine for a benchmark, but the story would be stronger if you add one or two domain baselines for each scenario (for example a simple reduced order model, or even a physics based filter) so readers have a calibration point.
- Throughout the paper there are small terminology issues. I would prefer "numerical error" over "computational error" (computational error sounds like a code bug). Be precise about whether errors arise from discretization, closure modeling, boundary conditions, or measurement.
- Not a criticism, but I think it would be better to rename the benchmark. "RealBench" seems far too broad and will clash with many domains. Something like "RealPDEBench" or "RealFlowBench" might be more appropriate.

### Questions
[Several questions discussed above]

### Soundness
2

### Presentation
2

### Contribution
2

### Rating
4

### Confidence
4

---

## Human Reviewer 2

### Summary
The paper describes a new physics-oriented benchmark and dataset that focuses on the gap between simulated and real experimental data. 

The paper presents $5$ applications and provides metrics for evaluation, alongside classical and recent benchmark methods. 

The description of the "TASK DEFINITION" should be improved and linked to the experiments, for example, in Table.1. 

The dataset is not well described; a summary table is necessary (how many samples, what is the size of each sample, what is the total real time duration, step size, dimension of the problem 1d, 2d, 3d, ...). 

The authors provide code to run the experiment, some sample data (real and simulated), but do not provide the script to generate the numerical data.

### Strengths
This dataset addresses the important aspect of bridging the gap between the simulated (numerical) and the physical system. 

While is not possible to cover a large experimental setup, the authors provide $5$ taks with both numerical and physical data.

This work will therefore help in evaluating new models, even if may not cover all possible scenarios.

### Weaknesses
It is hard to say, but it is not possible to cover all possible physical experimental conditions. Nevertheless, the paper is a good contribution in the right direction. 

The main point is that the numerical generation scripts are missing, therefore not possible to extend the data (at least numerical) to other scenarios. 

On the experimental side, I am not able to judge if the information is sufficient. 

I found the task description disconnected to the actual experiments. I would encourage the authors to improve that section.

### Questions
One possible source of difference between experimental and numerical experiments is the measurement noise, but i can assume there could be a larger difference, for example, if the state is not directly measurable (pressure is not available, but only velocity).

Could the author expand and position this paper in this context? What are the possible main and most critical differences between experiments and numerical simulations?

### Soundness
2

### Presentation
2

### Contribution
3

### Rating
6

### Confidence
3

---

## Human Reviewer 3

### Summary
The paper introduces RealBench, which evaluates the generalizability of simulated-data trained ML models on real-world measurements. It is well written, and covers an obvious gap in the literature. The authors clearly put significant thought into solving this issue, and it appears that the paper is as close to reproducible as is possible. II see they even posted their code on anonymous github. I looked through their code and it has the same quality/usability as PDEBench, which has set the standard for this kind of release.

### Strengths
The paper is original, and tackles a difficult challenge of pairing real and simulated data. This a clear gap in the literature, where simulated data was the only solution before. This provides a unique insight into many of the claims made on various architectures aimed at training surrogates for PDEs. It is comprehensive, well written, and thoughtfully formulated (especially the figures). It appears to achieve the maximum possible level of reproducibility through the use of the anonymous github repo.

### Weaknesses
The only weakness is the obvious one - out of domain regimes are not covered. However, this is probably the biggest area of weakness for this area of study as a whole. The complexity of fluid dynamics makes that a separate challenge entirely (one I dont see being solved any time soon). Surrogates typically cover some precise range of reynolds numbers around a specific geometry. It is the nature of this domain.

### Questions
I don't have any questions.

### Soundness
4

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4

---

## Human Reviewer 4

### Summary
This paper introduces a new dataset featuring paired simulation and experimental data for a variety of systems. The authors then benchmark a variety of state-of-the-art models on this data exploring the impact of simulated data on learning the real-world dynamics.

### Strengths
This type of dataset is so valuable to the community that I would strongly recommend acceptance even if the paper itself had major issues. Almost all work in this space today uses simulation data exclusively which is a space where learned models can at best offer speed improvements. This work and future datasets like it can offer a chance to explore in which regimes we may be able to achieve improved accuracy over numerical simulation and could great inform research directions over time.

That said, the paper itself is quite strong. Some particular reasons:
1. The presentation is excellent. It explains the problem very cleanly and describes prior work and the intended goal of this dataset.
2. The appendix is quite extensive providing data on the models and data generation process. In general, the division between main text and appendix content feels very well planned. 
3. Experimental fluids data is extremely rare and valuable to the community for developing models that can operate independently of the standard numerical solver framework.

### Weaknesses
The benchmarking has a few issues, but I'm largely reading this as a dataset paper, so I'm not marking down for them. However, in the interest of improving the paper, I will still list them out here.

Major:
1. The major missing component of this submission is an evaluation of how accurately the numerical simulation models the real scenario. If this is possible from the data, providing this information would drastically increase the potential impact of the paper. 
2. The model comparisons aren't very convincing due to the vastly different scales between them. It would improve the submission as a benchmarking effort to apply some form of normalization - FLOPs, parameter count, run time on fixed hardware - but I'm largely treating this as a dataset paper so not marking off for this. 
3. Similarly, one-step prediction is a useful reference for how well the models do what they're trained for, but longer rollout evaluations are more reflective of real tasks. I'd add more details on longer autoregressive rollouts if possible. 
4. Frequency domain error (in space) is generally going to be more informative when normalized. In 4.5 the text notes that a decrease of error in the high frequency is remarkable, but it's actually very much expected when comparing absolute metrics. Most fluid systems follow a polynomial decay law in spatial frequency, so the values being compared should be much smaller. It is actually concerning (for the baselines, not this submission) that this seems to be rare.

Minor notes
1. 3.1 prediction task - this is all discretized data, so it feels inaccurate to describe the task as a mapping between the continuous spaces. 
2. (paragraph beginning at 155) It would make sense to also include a canonical validation/test set of simulated data to evaluate the level of overfitting to simulation data and the overall sim2real gap. 
3. Table 4 is pretty hard to read. I don't know that this is the best way to demonstrate this data.

### Questions
1. Often one of the cited limitations of numerical simulation is the difficulty of simulating regimes (Reynolds/schmidt/ect) that occur commonly in real world settings. For CFD, the datasets currently contain mostly DNS data. Have you considered further comparisons between experimental data in regimes where DNS struggles and the more approximate simulations used to model them?
2. Do any of the models evaluated compare favorably to the numerical simulation for predicting the state of fields? Is the pairing sufficiently close that this is a question it is possible to ask with this data?

### Soundness
3

### Presentation
4

### Contribution
4

### Rating
10

### Confidence
4
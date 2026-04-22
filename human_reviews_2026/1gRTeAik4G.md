# Cyber-Zero: Training Cybersecurity Agents without Runtime

- Avg Score: 6.00
- Decision: Accept (Poster)
- Scores: 8, 8, 6, 2

## Abstract
Large Language Models (LLMs) have achieved remarkable success in software engineering tasks when trained with executable runtime environments, particularly in resolving GitHub issues. However, such runtime environments are often unavailable in other domains, especially cybersecurity, where challenge configurations and execution contexts are ephemeral or restricted. We present Cyber-Zero, the first runtime-free framework for synthesizing high-quality agent trajectories to train cybersecurity LLMs. Cyber-Zero leverages publicly available CTF writeups and employs persona-driven LLM simulation to reverse-engineer runtime behaviors and generate realistic, long-horizon interaction sequences without actual environments. Using trajectories synthesized by Cyber-Zero, we train LLM-based agents that achieve up to 13.1% absolute performance gains over baseline models on three prominent CTF benchmarks: InterCode-CTF, NYU CTF Bench, and Cybench. Our best model, Cyber-Zero-32B, establishes new state-of-the-art performance among open-weight models, matching the capabilities of proprietary systems like DeepSeek-V3-0324 and Claude-3.5-Sonnet while offering superior cost-effectiveness, and demonstrating that runtime-free trajectory synthesis can effectively democratize the development of state-of-the-art cybersecurity agents.

## Human Reviews

## Human Reviewer 1

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
*Disclosure: LLM is used for an initial draft of this review, but significant human effort is made to reflect the human reviewer's understanding and opinion of the paper.*

This paper addresses a current bottleneck in training large language model (LLM) agents for cybersecurity: the scarcity of high-quality, agentic training data. The authors note that unlike software engineering (SWE) where ample executable runtime environments are available to generate trajectories (e.g., from GitHub issues), cybersecurity environments are often ephemeral, restricted, or unavailable. This data scarcity has led to a significant performance gap between proprietary models and open-weight models.

To solve this, the paper introduces CYBER-ZERO, a novel and practical framework for synthesizing high-quality agent trajectories without needing an executable runtime. The core idea is to leverage publicly available Capture The Flag (CTF) writeups and simulate runtime environment there upon with a LLM. Specifically, CYBER-ZERO uses a "persona-driven dual-LLM" simulation to reverse-engineer the runtime behavior described in these writeups:
1. A "CTF Player" LLM (which does not see the solution) attempts to solve the challenge from scratch.
2. A "Bash Terminal" LLM (which does see the writeup and flag) acts as a "weak oracle," simulating realistic terminal outputs and providing minimal hints when the Player model gets stuck.

The result is a large-scale dataset (over 9,400 trajectories) that captures realistic, long-horizon interactions, including debugging and exploration. The authors use this dataset to fine-tune open-weight LLMs.

Models trained with CYBER-ZERO show significant performance gains (up to 13.1% absolute improvement) on three prominent CTF benchmarks. The best-trained open model, CYBER-ZERO-32B, achieves new state-of-the-art performance among open-weight models, matching more advanced models like Claude-3.5-Sonnet and DeepSeek-V3-0324.

### Strengths
- The paper tackles a well-defined, significant, and difficult problem: the data-scarcity bottleneck for training open-source cybersecurity agents. The proposed runtime-free framework, CYBER-ZERO, is a highly novel and pragmatic solution that effectively "upcycles" existing natural-language writeups into valuable, structured training data.
- The presented results are strong and convincing. The performance gains are clear and substantial across multiple models and benchmarks. The results model match the performance of top proprietary models while being far more cost-effective. Throughout ablations are also presented.
- The writing is clear and engaging.

### Weaknesses
- The quality of the generated traces depend greatly on the simulator LLM: it may need to, for example, simulate decompilation results and kernel exploits. Simulating these itself requires significant cybersecurity-related capabilities (not necessarily on the attack side, though), especially when the traces deviate from the writeups. While the shown experiment results are convincing, this may limit the upper bound of the method, as inaccurate traces may be less helpful for training.
- The authors state the hint mechanism is crucial for data collection, which may lead to the generated trajectories being guided by the writeups rather than authentic, self-driven exploration. This may limit the result LLMs' capability on discovering truly novel exploits.

### Questions
- Is the player LLM given the content of the writeup? The main body seems to suggest that it is not provided to the LLM, but the prompt provided in appendix B.2 mentioned the use of writeup.
- The performance of DeepSeek-V3-0324 on NYU CTF seems low compared to closed-weight systems. Curious if you have explanation on that.

### Soundness
3

### Presentation
4

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
This paper presents a novel approach of training LMs to be better Cybersecurity agents based on synthetic training trajectories. These training trajectories are generated using persona-driven LMs simulation without a real execution environment, i.e., one LM/agents impersonates the execution environment and simulates execution outputs, another LM/agent impersonates the agent/player. 

Experiments are performed based on the Qwen family, showing large gains on 8B, 14B, and 32B models. 
Analysis parts look at scaling with number of training trajectories and multiple rollouts (best@k).

### Strengths
Core contribution:

* Good motivation for why this work is important (scarcity of high-quality training data for cybersec applications, because execution environments are harder to obtain). 
* Very clearly data synthesis methodology which seems to be original. The data synthesis strategy seems to be very easy and applicable.
* The training dataset itself will also be a good asset and seems to be larger than previously available datasets. Clear demonstration of effectiveness of method, with large gains over the baseline models.
* Good analysis sections, looking at pass@k and also at scaling with number of training trajectories 
* Very smooth read.

Additional contributions:
  
* EnIGMA+ agent scaffolding improvements show good engineering and will help others use the scaffold. 
* Identification of broken benchmark instances also good contribution.

### Weaknesses
Minor comment:

The abstract states "Our best model, CYBER-ZERO-32B, [matches] the capabilities of proprietary systems like DeepSeek-V3-0324 and Claude-3.5-Sonnet", this seems to be an overstatement if one considers Fig 1 or Fig 3.

### Questions
What exact steps were taken to avoid contamination? The new dataset spans more than 1k challenges. The paper only says "manually removed 47 overlapping challenges", but doesn't go into details how these were identified. Since this is a crucial point for the validity of the results, some additional details or studies would be great to see (e.g., looking for overlaps in flag strings, or specific n-grams etc.). 

Presentational things I noticed/don't understand

* Fig 2: I am still confused by the arrows in the upper part of the figure. There's two personas, so what's the "LM" doing in the middle?  Or does this mean that the CTF challenge text is created based on the writeup with the help of the LM?

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
5

### Summary
This paper presents Cyber-Zero, a novel training data generation method that uses persona-based LLMs to produce realistic multi-turn trajectories of CTF task solving from human writeups without involving runtimes or actually executing commands. Cyber-Zero performs supervised  finetuning of open-source Qwen models and obtains noticeable percentage improvement, demonstrating that the generated trajectories are useful to improve cybersecurity performance of LLMs.

### Strengths
- The runtime-free framework to synthesize multi-turn trajectories from CTF writeups is an impactful technical contribution
- The paper provides comprehensive evaluation of multiple commercial and open-source LLMs on CTF benchmarks, and how their fine-tuning improves performance. Specifically, the result of SWE-Agent-LM not generalizing to CTF tasks is interesting
- The rectification of problematic challenges of existing CTF benchmarks is appreciated

### Weaknesses
- Aspects of the main contribution of synthesizing multi-turn trajectories need to be elaborated, such as how and when are hints provided. Please see "Questions" for specifics
- The paper lacks evaluation of whether this finetuning method has trained an LLM that overfits to a specific agentic format like EnIGMA. It raises pertinent questions about how the trained LLM would perform with a different agentic framework that is also targeted towards CTF tasks.
- Some aggregate statistics should be provided for the CTF writeups collected from CTFtime, to understand the composition of the finetuning data
- The contribution of EnIGMA+ seems insignificant and superfluous as it is not adequately discussed in the text, and it seems like a simple implementation change of deduplicating the docker port numbers

### Questions
- Table 1: It seems unnecessary to compare and contrast CVE-Bench, NYU CTF Bench, Cybench with Cyber-Zero as the former are benchmarks to evaluate LLM agents on cybersecurity tasks, while the later is a dataset for fine-tuning
- While Figure 2 gives an example of how synthetic trajectories are generated, some more details will be helpful to clarify the entire picture
	- How and when does the terminal model decide to provide hints to the player model?
	- How does the terminal model decide what hints to give?
	- If there is malformed, hallucinated, or unstructured outputs by either models, how is it handled?
- Line 144: provide text links of "CTFTime" and "CTF Archives" as footnotes instead of linking in the PDF
- Line 151: How does the DeepSeek model obtain any data that is missing from the CTF writeups, such as file contents?
- Line 247: Please provide details of the binary filter

### Soundness
3

### Presentation
3

### Contribution
4

---

## Human Reviewer 4

### Rating
2

### Rating Number
2

### Confidence
4

### Summary
This paper introduces Cyber-Zero, a framework for fine-tuning LLMs to perform better at CTFs. Specifically, they collect a large dataset of CTF writeups, synthetically generate trajectories, and then observe increased performance across three CTF datasets: InterCode, NYU, and Cybench.

### Strengths
The paper is well-organized and clear. The authors provide a helpful amount of detail and insight into their proposed method which cleverly uses a dual-LLM system to simulate agentic trajectories from previous CTF writeups that don't have accompanying runtime configurations.

### Weaknesses
This paper exclusively focuses on improving SOTA results of open source models on CTF benchmarks. The authors claim that Cyber Zero is a framework for training cybersecurity agents, when instead it is a framework for training CTF agents, just as ENIGMA is an agent scaffold strictly designed for CTFs and not for real world cybersecurity tasks. As a result, the usefulness of Cyber Zero is unproven. If the ENIGMA-scaffolded agents trained using Cyber Zero obtained state of the art results on more general benchmarks like MHBench, BountyBench, CVEBench, or XBOW's validation set, their claim of training cybersecurity agents would be more grounded. 

Further, while model performance using Cyber Zero does improve upon the baseline of open source models, this method does not reach absolute state-of-the-art. I am unconvinced that the ends justify the means for anyone who might want to deploy such a system, particularly because it is only "proven" on CTFs. 

The authors do not provide sufficient justification that their two-player synthetic generation pipeline is strictly required to achieve such results. For example, would single-shot trajectories yield similar results? And, relatedly, I am not convinced that simulating interactions in the fashion involved in this paper is a long-term worthwhile endeavor. Cybersecurity environments are no different from software engineering environments, and it is likelier that in the future environments with working runtimes that can scale will be ever-present. 

Finally, the authors make claims that their method is much more cost-effective than using API-based models like Claude Sonnet 3.7 that are able to achieve state of the art results, and yet they do not include fine-tuning or data generation cost in this analysis.

### Questions
1. How much did data generation cost? 
2. What was the total cost of fine-tuning these models on your hardware? 
3. Why did you choose to focus strictly on CTFs for this paper?

### Soundness
3

### Presentation
3

### Contribution
1

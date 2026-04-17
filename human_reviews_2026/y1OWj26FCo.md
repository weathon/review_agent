# Programming by Backprop: An Instruction is Worth 100 Examples When Finetuning LLMs

- Decision: Accept (Poster)
- Scores: 2, 6, 6, 6

## Abstract
Large language models (LLMs) are typically trained to acquire behaviours from demonstrations or experience, yet much of their training data is declarative: instructions, rules, and descriptions that specify behaviours without showing how to execute them. We introduce **Programming by Backprop (PBB)**: a training regime that enables LLMs to acquire *procedural* knowledge (i.e., reusable behaviours) from *declarative* instructions encountered during training. With PBB, instructions in training data provide an opportunity to "program" specific behaviours into model weights. The core principle underpinning PBB is the separation of learning how instructions map to behaviour from internalising new instructions. We devise two distinct PBB curricula that leverage this principle. Through controlled experiments across two domains (algorithmic execution from Python source code and text generation from context-free grammars), we demonstrate the benefit of these curricula over training on a homogeneous data mixture. Crucially, PBB is highly sample efficient, with *a single instruction substituting for up to 100 execution examples*. Though execution of instructions in training data remains less reliable than when instructions are given in-context, our results demonstrate that procedural knowledge can be noisily `programmed' into LLMs through PBB, with important implications for data curation and safety.

## Human Reviews

## Human Reviewer 1

### Rating
2

### Rating Number
2

### Confidence
3

### Summary
The paper studies whether LLMs can learn to execute "behaviors" from training on data that contains only their abstract description (as opposed to learning to execute them from demonstrations), which they call Programming By Backprop (BPP). The authors find that BPP does not emerge from standard pretraining. However, authors claim that BPP can be elicited with specific finetuning strategies.

### Strengths
This paper identifies the understudied phenomenon of learning to execute behaviours purely from abstract descriptions in the training data. I believe that understanding this phenomenon is potentially high-impact due to the usefulness of implicit behaviour learning in language models.  Thus, I find the significance of the studies to be the biggest strength of the paper.

### Weaknesses
It seems to me that the proposed methodology (Proactive PBB and Retroactive PBB) is not aligned with the paper's stated core hypothesis (line 189). **This makes the validity of the experiments difficult to understand in relation to the stated hypothesis.**
 
The core hypothesis only concerns a function whose abstract description is in the training data and that the model is expected to execute without training demonstrations. However, the actual method by which this hypothesis is tested involves training on data with demonstrations of related functions, which is absent in the statement of the hypothesis. Perhaps the hypothesis needs to be revisited to include dependency on training data from demonstrations of other functions.

I would encourage the authors to either extend the core hypothesis or otherwise precisely motivate why the specific experimental setup is a sound methodology to validate the stated hypothesis (given that the current experiments involve training on demonstration data, when the lack thereof is central to the hypothesis).

### Questions
0. **Why is the methodology a sound way to investigate the core BPP hypothesis, given the seeming mismatch in hypothesis and experiments described above?**
1. Training neural networks to execute code (or otherwise have programmatic behavior) is a well-stablished line of research (see e.g., [0-3]) that this manuscript, in my opinion, does not engage with enough. **I would encourage the authors to revisit this part of the literature and significantly expand the related work section.**
	- In particular, even though the BPP hypothesis (line 189) is cleanly stated and to my knowledge novel, the actual experiments are much more aligned with the standard "learning to program" problem due to the inclusion of demonstrations. This makes it particularly important to acknowledge and position this work relative to the aforementioned line of research.
2. A naive direct validation of the BPP hypothesis would *not* train on demonstration data at all. This follows from the statement of the BPP hypothesis. However, the methodology is not motivated and no experiments are reported at all under this "direct" setup. **Why is it necessary to use demonstration data? Why are there no experiments without demonstration data?**
3. The description of the RL setup is not described at enough detail.
	-  E.g., what is the reward function/reward model? How is the RL problem formulated?
4. A concern about the use of RL: **is the use of RL an effective way to validate the hypothesis of BPP?** Because in that case, the model wouldn't be trained solely on symbolic descriptions, thus would not validate the BPP hypothesis.
	- The experimental section would benefit from discussion of the RL experiment and how they relate to the original problem statement.
5. You state that "PBB [...] can be elicited through targeted finetuning strategies." These strategies are the CoT and RL results. However, there is no controlled experiment that tests CoT and RL *without* BPP. That is, e.g., how do you know that the performance gains are not only from CoT/RL?

[0] Zaremba et al, Learning to execute, 2014
[1] Tian et al, Learning to Infer And Execute 3D Shape Programs, 2019
[2] Yan, Neural Execution Engines: Learning to Execute Subroutines, 2020
[3] Waleed Gondal et al, Dynamic Inference with Neural Interpreters, 2021

### Soundness
1

### Presentation
2

### Contribution
2

---

## Human Reviewer 2

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
This paper explores the question of whether LLMs can be "programmed" by finetuning them on instances of symbolic descriptions of procedures without necessarily seeing their output. The phenomenon is termed "Programming by Backprop." The paper explores finetuning LLama and Qwen models on meta-learning paradigms in which they learn with both paired and unpaired examples of code (real and synthetic), ciphers, and grammars from which to generate text. Results are generally positive, showing in some cases that models can get better at executing unpaired programs seen during training.

### Strengths
S1. The paper is well-motivated and clearly written. It will be of interest to both computational linguists who study emergent behaviors, and potentially fine-tuning NLP practicioners who want to finetune models for algorithmic reasoning. 

S2. The paper explores a diverse array of kinds of data, from simple synthetic python programs to leetcode to ciphers and grammars.

S3. The ablations on data and stages answer a number of the questions that I had on first pass, indicating a substantial amount of analysis.

### Weaknesses
W1. The hypotheses on line 215 about the effect of the acquisition phase of Proactive and on 229 about the exposure phase of Retroactive could use evidence across datasets and models. 

Are they always doing something, or does the baseline that never sees the unpaired code until test time do just as well? Figure 4 Left and Figure 6 shows results for two task/model class combinations (Llama + Leetcode and Llama + Grammar), giving partial evidence supporting the acquisition phase, but unless I am missing something, I do not see the results for the Python code nor any results for Qwen. I also do not see results showing that the model learns anything during exposure to be retroactively taught during activation. A baseline that skips exposure would help.  This baseline would also help illustrate the core hypothesis, that the standard autoregressive training objective is what causes the model to internalise the executable representation of the procodure.

W2. Figure 2 is hard to interpret without the presence of untuned baselines for comparison. From train steps = 0 can we conclude that the LLMs have no ability to execute code prior to any PBB tuning? Even for simple programs?

W3. The choice to use only SFT for Proactive PBB but SFT+RL for Retroactive PBB seems somewhat arbitrary; the paper could benefit from explaining this design choice.

### Questions
Q1. The datasets use 100-500 unique code instances/grammars for training, but figure 9 shows that 800 programs can imrpove accuracy further. Was this maximum of 800 chosen arbitrarily, or does more than 800 perform about as well? 

Q2. How were hyperparameters in 4.2 chosen? Which were most impactful to the presence of the emergent behavior?

Q3. Python is one of the most prevelant languages in pretraining corpora. If you train on a programs in a less prominant language than Python, do the results generalize?

### Soundness
2

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
The paper describes how an LLM can be finetuned on a set of functions, with some of those functions having input/output examples, and it will learn to be able to compute input/output mappings for the functions that don't have input/output examples, when those functions are tested with inputs at test time they will execute correctly.
Essentially it memorizes all the functions during finetuning, and the finetuning on the input/output examples "teaches" the LLM how execute the functions so that the functions that don't have input/output pairs get correctly executed at test time on inputs.

The paper also show this finetuning process works on some other domains.

### Strengths
Honestly I was a bit surprised this approach worked at all when first reading the paper, I sort of assume test time access to the functions that had not been "executed" with input/output pairs at train/finetune time would be required, and that really this wouldn't work well outside of chain of thought sort of step by step walking through the functions line by line to compute the results at test time.  That functions presented at finetune time are memorized in an executable way is not what I would have expected.

Your approach is novel and unique and surprising that it works (to me) where how I would approach this problem is different (described below in weakness)

### Weaknesses
It feels a bit almost accidental that the way the LLM happens to encode the functions it has seen at finetune time without input/output pairs are able to "lean on" / "borrow" from the input/output pair experience of the functions that had input/output examples. It feels sort of hackey, the empirical results do show this transfer works, but it doesn't feel reliable to me.

I was unclear what the RL approach was from the paper, the SFT approach I think is more obvious what you would do at finetune time, but the RL approach is not so obvious, and I didn't see the code for it in my very brief search of the provided code.

I guess if I was to approach this problem, I would have done a COT sort of approach where at test time I would train the LLM to reproduce the complete function it's trying to execute in a thinking block, along with a chain of thought sort of scratchpad computation of interim results to execute the function, and then to close the thinking block and output the answer. I feel like that would give a more reliable chance of correctly executing the functions it saw with input/output pairs, and the functions that didn't have input/output pairs, just the LLM needs to remember the function and reproduce the function in a thinking block and execute it in a chain of thought-ish way.

But I guess your approach is novel and unique and surprising that it works (to me) where what I describe is maybe not so novel.

### Questions
Can you describe in more detail how the RL training was done?

### Soundness
3

### Presentation
3

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
This paper proposes TransferEngine, a portable RDMA-based point-to-point communication layer meant to support emerging LLM system patterns such as disaggregated inference (prefill/decoder split), large-scale asynchronous RL fine-tuning, and MoE dispatch/combine across heterogeneous NICs (NVIDIA ConnectX-7 and AWS EFA). The core motivation is that today’s LLM frameworks mostly rely on collectives (NCCL, torch.distributed), which assume fixed membership, ordered operations, and often uniform tensor shapes — assumptions that break down when you need dynamic scaling, sparse / per-token communication, or independent prefill/decoder pools

### Strengths
1. Timely problem & good problem framing. The paper is very in tune with where LLM serving is going: disaggregation (Splitwise, DistServe, Mooncake), MoE, and separate RL rollout fleets all want *non-collective*, *asymmetric*, *sometimes sparse* transfers. The authors clearly articulate where collectives fall short (fixed membership, ordering, shape uniformity), and that framing is convincing. 

2. Portability story is concrete, not hand-wavy. Most recent high-perf MoE / KV-store work quietly assumes ConnectX + IBGDA; this paper shows comparable latency on ConnectX-7 and a working EFA path, including notes on libfabric quirks and the need to aggregate 4×100 Gbps EFAs to reach 400 Gbps. That’s a real engineering pain point in production clouds, so demonstrating it is valuable. 

3. Case studies are well chosen. KV-cache streaming, 1.3-second RL weight pushes for trillion-parameter models, and MoE dispatch/combine basically cover the three hardest communication patterns people are actually building right now. This makes the paper much more than a “we wrapped libibverbs + libfabric” story. 

4. Performance numbers are credible and reasonably detailed. They compare against `ib_write_bw` / `fi_rma_bw`, show saturation thresholds, and offer side-by-side with NIXL where available. The MoE section even breaks out send vs receive and shows where the proxy overhead lands. This is the kind of evidence that’s often missing in “we built a new transport” papers.

### Weaknesses
1. Related-work positioning is a bit soft. The authors mention NVSHMEM, NIXL, Mooncake’s RDMA transfer engine, etc., but they stop short of a head-to-head systems comparison on all of them in the *same* setting. In particular, NIXL has begun adding EFA; NVSHMEM has both GPU-initiated and host-proxy modes; Mooncake targets KV-centric serving. A tighter comparison table (“who supports EFA,” “who assumes ordering,” “who can do MoE dispatch”) would make the novelty sharper. Right now the contribution can be read as “we unified the lowest common denominator and polished the proxy path,” which is good engineering but slightly incremental. 

2. Evaluation breadth is narrower than it could be. Most experiments are on 8×H200 nodes with either CX-7 or 2×200 Gbps EFA. That’s a fairly high-end, clean setup. There’s no stress test on less symmetric topologies, multi-tenant noise, or mixed NIC generations — all scenarios cloud users actually hit. This weakens the “portable across providers” claim a bit: we see two well-supported targets, not the messy real world. (This is analogous to “dataset not enough, weakens the conclusion’s solidity” in your template.)

3. Limited ablations on the core idea (IMMCOUNTER + unordered). The paper asserts that giving up on ordering and pushing completion into a counter lets them unify EFA and CX-7, but it doesn’t really show: (a) the overhead of the counter path vs relying on RC ordering, (b) how often counters get out of sync under loss / retries, or (c) how the callback thread scales when 56 peers are all enqueuing WRITEs in the MoE case. Since this is the distinctive technical idea, I’d like one figure that says “without IMMCOUNTER on EFA, this MoE test breaks / slows to X; with it, we get Y.”

### Questions
None

### Soundness
3

### Presentation
3

### Contribution
3

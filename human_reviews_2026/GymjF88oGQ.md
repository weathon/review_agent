# The Pensieve Paradigm: Stateful Language Models Mastering Their Own Context

- Avg Score: 5.50
- Decision: Accept (Poster)
- Scores: 6, 4, 4, 8

## Abstract
In the world of Harry Potter, when Dumbledore's mind is overburdened, he extracts memories into a Pensieve to be revisited later. In the world of AI, while we possess the Pensieve—mature databases and retrieval systems, our models inexplicably lack the ``wand'' to operate it. They remain like a Dumbledore without agency, passively accepting a manually engineered context as their entire memory.
This work finally places the wand in the model's hand. We introduce StateLM, a new class of foundation models endowed with an internal reasoning loop to manage their own state. We equip our model with a suite of memory tools, such as context pruning, document indexing, and note-taking, and train it to actively manage these tools. By learning to dynamically engineering its own context, our model breaks free from the architectural prison of a fixed window.
Experiments across various model sizes demonstrate StateLM's effectiveness across diverse scenarios. On long-document QA tasks, StateLMs consistently outperform standard LLMs across all model scales; on the chat memory task, they achieve absolute accuracy improvements of 10% to 20% over standard LLMs. On the deep research task BrowseComp-Plus, the performance gap becomes even more pronounced: StateLM achieves up to 52% accuracy, whereas standard LLM counterparts struggle around 5%. Ultimately, our approach shifts LLMs from passive predictors to state-aware agents where reasoning becomes a stateful and manageable process.

## Human Reviews

## Human Reviewer 1

### Rating
6

### Rating Number
6

### Confidence
4

### Summary
The paper introduces **The Pensieve Paradigm**, a framework that equips an LLM with tools to dynamically manage its own context and reasoning process. The proposed system, called **StateLM**, allows the model to actively construct, prune, and update its working memory through operations such as dynamic indexing, note-taking, and context deletion. This enables it to go beyond the predefined context length and handle extremely long sequences efficiently.

The authors propose a data generation pipeline, and employ SFT to train models from the Qwen3 family. Empirical results show that StateLM achieves strong performance in both synthetic and real-world tasks, consistently outperforming the baselines, while using only a fraction of the context length.

### Strengths
- The paper is well written, and the main ideas are presented clearly.
- I believe that the problem that the authors tackle is both interesting, and useful, particularly in the context of agentic systems. In many cases, agents are required to handle huge contexts (for instance large code repos); therefore, the proposed paradigm could be leveraged to both improve the performance and make agentic systems more efficient.
- The paper has solid amount of empirical results that showcase the strength of the proposed method, covering both synthetic and real-world settings.

### Weaknesses
- As the authors also mention, the current set of tools is predefined and fixed. While this appears sufficient for the evaluated tasks (Needle-in-a-Haystack and Long Document QA), it may prove inadequate in more complex or dynamic scenarios
- In Section 4.1, I think that the comparison with the baseline may not be entirely fair. The high accuracy achieved by the proposed approach on extremely long contexts is indeed interesting;  however, the poor performance of the baseline models in the cases where the context length is exceeded is not surprising. Perhaps, a better baseline could be to use a simple sliding window approach, where the size of the window is close to the context size of the model.
- Although the authors note this as well, I believe that testing RL approaches would be reasonable in this setting, since there has been evidence that RL works well in similar settings(e.g [1]). 

[1] Feng, Jiazhan, et al. "Retool: Reinforcement learning for strategic tool use in llms." arXiv preprint arXiv:2504.11536 (2025).

### Questions
- In Figure 5, why does the inference time of  Qwen3-8B (baseline) decrease as the context length increases? Is it due to truncation?

### Soundness
3

### Presentation
3

### Contribution
3

---

## Human Reviewer 2

### Rating
4

### Rating Number
4

### Confidence
4

### Summary
Pensieve/StateLM reframes long-context modeling as learned memory management. The LLM executes a tool-use loop ( e.g., analyzeText, buildIndex, searchEngine) to construct and prune its own working context rather than passively consuming a fixed window. StateLM is trained on Claude Sonnet 4–generated trajectories filtered for outcome and behavior on NovelQA and NarrativeQA. On Needle-in-a-Haystack (NIAH), StateLM outperforms Qwen-3 instruct baselines (4B/8B). On NovelQA and InfiniteBench, a 32K-context StateLM surpasses a 128K instruct baseline and shows scaling trends with higher inference-time compute. A prompt-only agent underperforms StateLM, supporting the claim that these behaviors should be learned, not merely prompted.

### Strengths
- The data curation pipeline is carefully designed. multi-stage filtering and process-mode classification (search vs. scan) produce cleaner trajectories for training.
- SLM w/o search greatly outperforms baseline by a large margin especially after 256K tokens.
- On real-world tasks like NovelQA and InfiniteBench, the results are impressive where SLM with short context (32K) can achieve better performance than instruct model with context of 128K token
- Good writing. The description of StateLM and experiments are clear and easy to follow.

### Weaknesses
- It is not clear how StateLM materially differs from prior work (e.g., A-Mem, SCM, Dynamic Cheatsheet). The claim of “not a fixed workflow loop” does not really establish novelty, as this function has been supported by agentic toolkit like Anthropic’s Model Context Protocol (MCP) and also has been explored by prior work.
- The training trajectories come from Sonnet-4, which along with many open-source agents already can decide which tools to use given context. As presented, the contribution is largely policy learning over a fixed toolset (index/search/read/note/delete), rather than a new memory paradigm.
- Ablations are limited: comparisons are mainly against a prompt-only baseline. There are no per-tool ablations (e.g., removing deleteContext) and no robustness analysis under noisy/failed tool calls.
- All results are based on only two models Qwen3 4B/8B of the same family. It is not clear if it generalizes well to other families.
- StateLM appears to use substantially more inference-time interaction/compute than single-pass baselines, making the comparison potentially not apples-to-apples.

References:

```
@article{wang2024openhands,
  title={Openhands: An open platform for ai software developers as generalist agents},
  author={Wang, Xingyao and Li, Boxuan and Song, Yufan and Xu, Frank F and Tang, Xiangru and Zhuge, Mingchen and Pan, Jiayi and Song, Yueqi and Li, Bowen and Singh, Jaskirat and others},
  journal={arXiv preprint arXiv:2407.16741},
  year={2024}
}
```

```
@article{xu2025mem,
  title={A-mem: Agentic memory for llm agents},
  author={Xu, Wujiang and Mei, Kai and Gao, Hang and Tan, Juntao and Liang, Zujie and Zhang, Yongfeng},
  journal={arXiv preprint arXiv:2502.12110},
  year={2025}
}
```

```
@article{yu2025memagent,
  title={MemAgent: Reshaping Long-Context LLM with Multi-Conv RL-based Memory Agent},
  author={Yu, Hongli and Chen, Tinghong and Feng, Jiangtao and Chen, Jiangjie and Dai, Weinan and Yu, Qiying and Zhang, Ya-Qin and Ma, Wei-Ying and Liu, Jingjing and Wang, Mingxuan and others},
  journal={arXiv preprint arXiv:2507.02259},
  year={2025}
}
```

```
@article{wang2023enhancing,
  title={Enhancing large language model with self-controlled memory framework},
  author={Wang, Bing and Liang, Xinnian and Yang, Jian and Huang, Hui and Wu, Shuangzhi and Wu, Peihao and Lu, Lu and Ma, Zejun and Li, Zhoujun},
  journal={arXiv preprint arXiv:2304.13343},
  year={2023}
}
```

### Questions
1. Do you report a baseline using Qwen3 + MCP tooling for different tasks? Since Qwen3 family natively support building AI agents with MCP protocols, the prompt-only ablation may underserve tool use capability of instruct model.
2. Do you provide error analysis for NIAH, NovelQA, and InfiniteBench for both StateLM and baselines (instruct and prompt-based)?
3. What is the impact of each tool (e.g., removing deleteContext) on accuracy? Can you provide further ablation analysis?

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 3

### Rating
4

### Rating Number
4

### Confidence
3

### Summary
This paper presents StateLM, a stateful, memory-augmented LLM agent supervisedly trained to
autonomously manage its own context through a (predefined) set of tools. Instead of relying on a
human-defined workflow, the model learns, via SFT on author-curated reasoning trajectories, to
decide when to perform indexing, note-taking, searching, and pruning. Experiments on long-
context QA benchmarks, compared with Qwen3-Instruct models, demonstrate substantial gains
in both efficiency and scalability.

### Strengths
● The problem addressed in this paper is crucial: transitioning from stateless LLMs to a
stateful paradigm enables long-term reasoning, multi-turn dialogue memory, and
cross-session continuity.
● The paper is well-written, and the case study in Section 3 provides an intuitive and
effective way to illustrate the Pensieve paradigm.
● The “model as the wizard” framing, i.e., pushing the model toward fully autonomous
decision-making about when (and, potentially in future work, how) to manage its own
context, is an appealing selling point that makes the work conceptually engaging.
However, as discussed below in the weaknesses, it remains to be seen whether this
setting can match the performance of prior semi-automated workflows.

### Weaknesses
1. The paper aims at a meaningful goal of achieving a fully automated workflow, since
heuristic and human-defined pipelines may not fully unlock the capability of LLMs.
However, the framework still relies on manually defined tools, making it essentially
semi-automated. Given that prior work (e.g., Memory-R1) also trains models to learn
what memory operations to perform, the main difference here seems to lie in when
those operations are triggered. Memory-R1 updates memory after each turn, which is
a natural and reasonable design, while this work lets the model decide the timing in
an end-to-end way.
○ The point is that whether this flexibility is actually an advantage is not obvious. 
I would like to see stronger empirical evidence, for
example, the proposed method triggers memory operations much less
frequently and therefore achieves higher efficiency, better utility, or broader
generalization across question types/domains to justify this design choice.
2. **Limited benchmark and model scope:** Evaluation is conducted only on a single
base model (Qwen3-Instruct) and two document-based QA datasets (NovelQA and
infiniteBench En.MC split). Broader long-term or multi-session benchmarks such as
LoCoMo, MSC, LongMemEval, or RULER (not necessarily all) should be included to
test richer memory behaviors.
3. **Lack of scalability and generalization tests:** The model is trained on the
*PublicDomain* split of NovelQA and mainly evaluated on the *Copyright* split. In
Table 3, its gains diminish on $\infty$Bench, particularly for larger backbones (e.g.,
Qwen3-8B). Additional experiments on cross-domain or OOD settings are needed to
assess the proposed method’s applicability, especially given its reliance on formatted
training.
4. **Missing baselines:**
○ Since the method equips the LLM with external memory, a fair comparison
should include the same base model with (i) direct function-call memory
access and (ii) MCP-based memory access (e.g.,
https://github.com/doobidoo/mcp-memory-service), to see whether the base
model (without SFT on reasoning traces) can already use memory when
given access, and how much extra benefit the proposed framework actually
provides.
○ Comparisons with prior memory-management methods, whether retrieval-
augmented (RAG) or agent-based methods such as those evaluated in Mem0
and Memory-R1, are necessary; For cost reasons, even a direct evaluation
within their setups would make the results more informative.

### Questions
5. What are the specific requirements for the “high-quality, good-behavior, expert
reasoning trajectories”?
6. Since the generation of expert reasoning trajectories is effectively a distillation
process from Claude-Sonnet-4, it would be informative to include Claude-Sonnet-4’s
own performance as a baseline in the Experimental section.

### Soundness
3

### Presentation
3

### Contribution
2

---

## Human Reviewer 4

### Rating
8

### Rating Number
8

### Confidence
3

### Summary
Stateful Language Models (StateLM or SLM) is a class of foundation models that are equipped with tools (including: dynamic indexing, context pruning, note taking) to manipulate their state in a reasoning loop which dynamically (and automatically) updates their context. A StateLM (i) can retrieve a needle in 1 million-token haystack (ii) in empirical results over practical QA benchmarks it performs better than strong instruct baselines using a fourth of their active context and (iii) is superior in learning to manage memory than agent-like prompting.

SLM's reasoning trajectory consists of a series of actions (thoughts and acts (tool invocations)) and states (optionally modified contexts and responses from tools): tools can do tasks like analyze text, summarize or search through it or even update (e.g. delete) context. SLM is trained over trajectories for handling questions of types involving either locating or understanding text (search or scan types): each trajectory consists of steps (training samples) where given the history up to some step, SLM is trained to predict next step's thought and action. Interestingly, SLM's performance cannot be matched by models that have access to tools and are prompted to follow the context management process in SLM.

### Strengths
- This is a simple and novel idea: the model becomes active, inspects its current memory/state and accordingly constructs the context to operate on using pre-defined tools.

- No pressing need for the user-in-the-middle role of building prompts conditioned on a manually inspected state (automation).

- Clean guidelines for training, set of orthogonal tools well-defined.

- Performance on long-context recall and QA benchmarks are impressive.

### Weaknesses
- Critical requirement for the availability of a strong LLM for the generation of training samples (in particular for process-mode classification)
- The set of tools is given, is generic enough but it certainly cannot fit any question handling.

### Questions
- Are there any thoughts for automatically building the set of tools most amenable to particular question types?

- Can the succession of the particular tools invoked drive the classification of questions answered into finer-grained classes?

### Soundness
3

### Presentation
4

### Contribution
4

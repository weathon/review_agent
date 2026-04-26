# source ../.env
# export ANTHROPIC_API_KEY=""
# export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
# export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
# export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
# export NEUTRAL_MODEL="glm-5.1"
# export CALIBRATION_SET="2026"
# export OUTPUT_CSV="bench_scores_claude_sonnet_2026.csv"
# export MERGE_LOG="pipeline_whole_claude_2026.log"
# ollama serve & 
# # use balanced to see quality in the whole spectrum of scores
# export CONCURRENCY=2    
# # python main.py --n_samples 100 --benchmark ../iclr2025/ --seed $(cksum <<< 'who 3r324f34 you' | cut -f 1 -d ' ') 
# # python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 300 --benchmark ../iclr2025/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 

# # python main.py --n_samples 200 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '45678765456787654567' | cut -f 1 -d ' ') 

source ../.env
export ANTHROPIC_API_KEY=""
export OPENAI_DEFAULT_MODEL="z-ai/glm-5.1"
export HARSH_MODEL="claude_sdk:claude-sonnet-4-6"
export MERGER_MODEL="claude_sdk:claude-sonnet-4-6"
export NEUTRAL_MODEL="ollama:glm-5.1:cloud"
export OUTPUT_CSV="bench_scores_claude_sonnet.csv"
export CALIBRATION_SET="2026"
export MERGE_LOG="pipeline_whole_claude_3.log"
# export MAX_PAPERS=100
ollama serve & 
# use balanced to see quality in the whole spectrum of scores
export CONCURRENCY=1
python main.py --n_samples 100 --benchmark ../iclr2026_cspaper/ --seed $(cksum <<< '早上闹钟响了我没鸟它，再响我还是没鸟它，第三次响的时候我直接把它创飞了告诉自己再睡五分钟结果一觉睡到九点差一刻，醒来盯着天花板开始哲学三连：我是谁我在哪我昨晚为什么要刷短视频刷到凌晨两点，思考完发现没有答案于是不思考了，爬起来刷牙牙膏空了我对着管子又挤又捏又揉搞得跟给牙膏做心肺复苏一样最后挤出来个黄豆大小的东西，洗脸水是冰的我整个人当场觉醒激灵得像被前任发的消息冻醒，照镜子头发支棱得跟我昨天的代码一样毫无章法但是反正也没人看就这样吧，走到厨房打开冰箱里面躺着两个鸡蛋半根已经在思考人生的黄瓜还有一盒昨天的外卖在那里幽幽地看着我，我关上冰箱又打开冰箱又关上又打开重复了大概四次仿佛冰箱里会突然长出一份满汉全席，最后认命了泡面，烧水的时候我开始刷手机然后水烧干了锅在那嗷嗷叫我才反应过来卧槽，重新烧水这次我死死盯着它结果watched pot never boils这破水烧得比我导师回邮件还慢，我又开始刷手机面终于泡好了吃第一口给我咸出一个激灵感觉我刚刚不是在吃面是在喝海，但是吃都吃了硬着头皮造完，吃完看时间快十点了我深沉地叹了一口气看着水池里的碗那个碗看着我我看着它我俩沉默对视了大概十秒最后我说算了哥们改天吧然后转身走了。' | cut -f 1 -d ' ') 
# python main.py --n_samples 300 --benchmark ../iclr2026/ --seed $(cksum <<< '🍍' | cut -f 1 -d ' ') 
# python main.py --n_samples 200 --benchmark ../iclr2026/ --seed $(cksum <<< '23456789876543234567' | cut -f 1 -d ' ') 

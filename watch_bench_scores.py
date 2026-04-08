import os, subprocess, sys, time, wandb
csv="bench_scores.csv"; img="bench_scores_scatter.png"; last=None; run=wandb.init(project=os.getenv("WANDB_PROJECT","review-agent-bench"))
while True:
    try: cur=open(csv,"rb").read()
    except FileNotFoundError: cur=None
    if cur!=last and cur is not None: subprocess.run([sys.executable,"metric.py",csv],check=False); os.path.exists(img) and run.log({"bench_scores_scatter": wandb.Image(img)}); last=cur
    time.sleep(10)
do

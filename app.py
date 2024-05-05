from src.myTrain import Tranner
mid = 'unsloth/mistral-7b-instruct-v0.2-bnb-4bit'
dataSrc='/root/autodl-tmp/DISC-Law-SFT-Pair.jsonl'
import subprocess
import os
result = subprocess.run('bash -c "source /etc/network_turbo && env | grep proxy"', shell=True, capture_output=True, text=True)
output = result.stdout
for line in output.splitlines():
    if '=' in line:
        var, value = line.split('=', 1)
        os.environ[var] = value
my_trainer = Tranner(mid,dataSrc)
my_trainer.loadModel()
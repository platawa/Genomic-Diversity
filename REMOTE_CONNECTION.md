# Remote Server Connection Guide

## Cluster Details
- **Host:** `orcd-login001.mit.edu`
- **User:** `platawa`
- **Remote project dir:** `/orcd/data/zhang_f/001/platawa/jan31_files/`
- **Conda env:** `evo2_sep28`

## SSH Connection via ControlMaster

To avoid repeated Duo MFA pushes, this project uses an SSH ControlMaster socket.
**Prerequisite:** You must have an active SSH session open in another terminal.

### Check if socket is alive
```bash
ssh -o ControlPath="$HOME/.ssh/platawa@orcd-login001.mit.edu:22" \
    -o ControlMaster=no -O check platawa@orcd-login001.mit.edu 2>/dev/null
```

### Run a remote command through the socket
```bash
ssh -o ControlPath="$HOME/.ssh/platawa@orcd-login001.mit.edu:22" \
    -o ControlMaster=no platawa@orcd-login001.mit.edu "COMMAND"
```

### Helper function
```bash
source scripts/ssh_connect.sh
remote_cmd "ls -la /orcd/data/zhang_f/001/platawa/jan31_files/"
```

## Syncing Files to Cluster
```bash
rsync -avz --exclude '__pycache__' \
    tools/analyze_scoring_results.py \
    platawa@orcd-login001.mit.edu:/orcd/data/zhang_f/001/platawa/jan31_files/tools/
```

## Running on Cluster Compute Nodes

The user typically allocates interactive SLURM sessions:
```bash
# Check allocated jobs
squeue -u platawa

# SSH to an allocated node (from login node)
ssh node4102

# Activate environment
conda activate evo2_sep28
cd /orcd/data/zhang_f/001/platawa/jan31_files/
```

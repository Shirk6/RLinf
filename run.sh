srun --jobid=450046 --overlap -w dgx-18 -N1 -n1 --gres=gpu:8 \
/cm/local/apps/apptainer/current/bin/apptainer exec --nv \
--bind /project/peilab/srk/rss_2026_ws:/project/peilab/srk/rss_2026_ws \
/project/peilab/srk/.cache/enroot/rlinf-embodied-wan-openpi-shirk6.sif \
bash -lc 'cd /project/peilab/srk/rss_2026_ws/RLinf && bash examples/embodiment/run_embodiment.sh wan_battery_pi05_grpo'
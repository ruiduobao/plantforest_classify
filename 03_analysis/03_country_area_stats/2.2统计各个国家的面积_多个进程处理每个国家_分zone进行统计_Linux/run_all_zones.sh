#!/bin/bash
SCRIPTS_DIR="/work/home/chengrui1075/SEA_TREE/CODE/3.analy/calarea/scripts"
LOG_DIR="/work/home/chengrui1075/SEA_TREE/CODE/3.analy/calarea/log"
CODE_DIR="/work/home/chengrui1075/SEA_TREE/CODE/3.analy/calarea"

mkdir -p "$SCRIPTS_DIR" "$LOG_DIR"

CTL_LOG="$LOG_DIR/run_all_zones_submit.log"
echo "[$(date +%F_%T)] START submissions" >> "$CTL_LOG"

for zone in {1..10}; do
  for year in {2017..2024}; do
    jobname="zone${zone}_${year}"
    script="$SCRIPTS_DIR/cal_area_zone${zone}_${year}.sh"
    cat > "$script" <<EOF
#!/bin/bash
#SBATCH --job-name=${jobname}
#SBATCH --output=${LOG_DIR}/${jobname}_%j.out
#SBATCH --error=${LOG_DIR}/${jobname}_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=40
#SBATCH --time=24:00:00
#SBATCH --partition=tyhcnormal
source ~/.bashrc
conda activate geodetector
cd ${CODE_DIR}
python ./2.2.cal_ZONE_NUMBER_LINUX.py --zone ${zone} --year ${year}
EOF
    chmod +x "$script"
    tries=0
    while true; do
      submit_out=$(sbatch "$script" 2>&1)
      rc=$?
      if [ $rc -eq 0 ]; then
        jobid=$(echo "$submit_out" | awk '{print $NF}')
        nodeinfo=$(squeue -j "$jobid" -o "%i %t %N" 2>/dev/null | tail -n 1)
        echo "[$(date +%F_%T)] SUBMIT ${jobname} id=${jobid} ${nodeinfo}" >> "$CTL_LOG"
        break
      else
        tries=$((tries+1))
        echo "[$(date +%F_%T)] RETRY ${jobname} try=${tries} msg=${submit_out}" >> "$CTL_LOG"
        if [ $tries -ge 3 ]; then
          echo "[$(date +%F_%T)] FAIL ${jobname} after ${tries} tries" >> "$CTL_LOG"
          break
        fi
        sleep 2
      fi
    done
  done
done
echo "[$(date +%F_%T)] DONE submissions" >> "$CTL_LOG"
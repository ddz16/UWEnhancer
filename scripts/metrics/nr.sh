#!/usr/bin/env bash

# dir="FUnIEGAN"

# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/C60" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/LSUI" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/OceanDark" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/T90" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/U45" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/UIID" >> "results/$dir/log.txt"



dir="01_UConvNeXt_LSUI_GlobalBranch_test/visualization"

python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/C60" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/LSUI" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/OceanDark" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/T90" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/U45" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/UIID" >> "results/$dir/log.txt"

dir="01_UConvNeXt_LSUI_LocalBranch_test/visualization"

python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/C60" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/LSUI" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/OceanDark" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/T90" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/U45" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/UIID" >> "results/$dir/log.txt"

dir="01_UConvNeXt_LSUI_test/visualization"

python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/C60" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/LSUI" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/OceanDark" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/T90" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/U45" >> "results/$dir/log.txt"
python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/UIID" >> "results/$dir/log.txt"



# dir="PUGAN"

# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/C60" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/LSUI" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/T90" >> "results/$dir/log.txt"
# python scripts/metrics/calculate_all_nr_metrics.py --input "results/$dir/U45" >> "results/$dir/log.txt"
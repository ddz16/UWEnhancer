#!/usr/bin/env bash

# dir="Fusion"
# rm results/Fusion/log_fr.txt
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"


# dir="ICSP"
# rm results/ICSP/log_fr.txt
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"


# dir="FUnIEGAN"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"


# dir="UGAN"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"


# dir="PUGAN"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
# python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"

dir="01_UConvNeXt_LSUI_GlobalBranch_test/visualization"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"


dir="01_UConvNeXt_LSUI_LocalBranch_test/visualization"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"

dir="01_UConvNeXt_LSUI_test/visualization"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/LSUI/GT" --restored "results/$dir/LSUI" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIEB/reference-890" --restored "results/$dir/T90" >> "results/$dir/log_fr.txt"
python scripts/metrics/calculate_all_fr_metrics.py --gt "datasets/UIID/GT" --restored "results/$dir/UIID" >> "results/$dir/log_fr.txt"

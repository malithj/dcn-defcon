#!/bin/bash
source venv/bin/activate
echo -e "compiling normal mode"
rm -rf build
rm -rf dcn_ext.cpython* 
rm -rf Deform.egg-info/
python setup.py develop
python exported_model/model_driver.py 
cp results/$(ls -Atr results | tail -n 1) output/model_deformable_runtimes_nvcc.csv
python exported_model/auto_run_model.py
cp results/$(ls -Atr results | tail -n 1) output/model_deformable_runtimes.csv
python driver.py
cp results/$(ls -Atr results | tail -n 1) output/deformable_runtimes.csv
python profile_driver.py
cp results/$(ls -Atr results | tail -n 1) output/profile_info_deformable_runtimes.csv

echo -e "compiling low resolution mode"
rm -rf build
rm -rf dcn_ext.cpython* 
rm -rf Deform.egg-info/
python setup_lowres.py develop
python exported_model/model_driver.py
cp results/$(ls -Atr results | tail -n 1) output/model_deformable_runtimes_nvcc_low_res.csv
python exported_model/auto_run_model.py
cp results/$(ls -Atr results | tail -n 1) output/model_deformable_runtimes_low_res.csv
python driver.py
cp results/$(ls -Atr results | tail -n 1) output/deformable_runtimes_low_res.csv
python profile_driver.py
cp results/$(ls -Atr results | tail -n 1) output/profile_info_deformable_runtimes_low_res.csv


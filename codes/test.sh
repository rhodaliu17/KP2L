gpu='0'

python test.py --gpu $gpu \
                     --data_dir '../data/NIH' \
                     --list_dir '../datalist/Pancreas' \
                     --load_path $load_path
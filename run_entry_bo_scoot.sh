model_path=$1
model_name=$2
dataset_path=$3
dataset_name=$4
request_rate=$5
request_num=$6
gpu_num=$7
gpu_type=$8

# install requirements
pip install -r requirements.txt

# obtain the default tp, and update max_sequence_length
cd tuner_conf
bash tuner_conf.sh ${model_path}
cd ..


echo submit.sh
echo model_path=${model_path}
echo model_name=${model_name}
echo dataset_path=${dataset_path}
echo dataset_name=${dataset_name}
echo request_rate=${request_rate}
echo request_num=${request_num}
echo gpu_num=${gpu_num}
echo gpu_type=${gpu_type}


python bo_scoot.py --model_path ${model_path}\
                    --dataset_path ${dataset_path}\
                    --dataset_name ${dataset_name}\
                    --model ${model_name}\
                    --total_resource ${gpu_num}${gpu_type}_mobo\
                    --request_rate ${request_rate}\
                    --bo_loop 30\
                    --exp_num 1\
                    --num_requests ${request_num}\
                    --num_obj 3\
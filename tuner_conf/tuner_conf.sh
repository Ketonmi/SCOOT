model_path=$1

echo tuner_conf.sh
echo python tuner_conf.py --model_path ${model_path} 

python tuner_conf.py --model_path ${model_path}

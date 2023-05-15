device_idx=${1:-0}
epochs=300
batch_size=128
lr=0.005
weight_decay=0.001
noise_prob=${2:-0.4}

echo use_device:${device_idx}

CUDA_VISIBLE_DEVICES=${device_idx} python3 main.py \
--epochs ${epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--weight_decay ${weight_decay} \
--noise \
--noise_rgb \
--noise_prob  ${noise_prob} \
--ZCA \
--batch_level_zca
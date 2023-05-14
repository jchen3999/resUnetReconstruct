device_idx=${1:-0}
epochs=400
batch_size=256
lr=0.0001
weight_decay=0.0001

echo use_device:${device_idx}

CUDA_VISIBLE_DEVICES=${device_idx} python3 main.py \
--epochs ${epochs} \
--batch_size ${batch_size} \
--lr ${lr} \
--weight_decay ${weight_decay}
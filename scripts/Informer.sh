model_name="Informer"

for i in 10 15 20 25 30
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate/ \
        --data_path oisst_lat_14.0_lon_112.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --label_len $((2 * $i)) \
        --pred_len $i \
        --enc_in 11 \
        --dec_in 11 \
        --c_out 11 \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 2048 \
        --dropout 0.1 \
        --train_epochs 10 \
        --batch_size 32 \
        --patience 3 \
        --des "20240601" \
        --lradj "HalfLR" \
        --learning_rate 0.0001 \
        --model_id "omdata1_test1" \
        --inverse
done

for i in 10 15 20 25 30
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate/ \
        --data_path oisst_lat_14.0_lon_118.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --label_len $((2 * $i)) \
        --pred_len $i \
        --enc_in 11 \
        --dec_in 11 \
        --c_out 11 \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 2048 \
        --dropout 0.1 \
        --train_epochs 10 \
        --batch_size 32 \
        --patience 3 \
        --des "20240601" \
        --lradj "HalfLR" \
        --learning_rate 0.0001 \
        --model_id "omdata2_test1" \
        --inverse
done

for i in 10 15 20 25 30
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate/ \
        --data_path oisst_lat_20.0_lon_112.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --label_len $((2 * $i)) \
        --pred_len $i \
        --enc_in 11 \
        --dec_in 11 \
        --c_out 11 \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 2048 \
        --dropout 0.1 \
        --train_epochs 10 \
        --batch_size 32 \
        --patience 3 \
        --des "20240601" \
        --lradj "HalfLR" \
        --learning_rate 0.0001 \
        --model_id "omdata3_test1" \
        --inverse
done

for i in 10 15 20 25 30
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate/ \
        --data_path oisst_lat_20.0_lon_118.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --label_len $((2 * $i)) \
        --pred_len $i \
        --enc_in 11 \
        --dec_in 11 \
        --c_out 11 \
        --e_layers 2 \
        --d_layers 1 \
        --d_model 512 \
        --d_ff 2048 \
        --dropout 0.1 \
        --train_epochs 10 \
        --batch_size 32 \
        --patience 3 \
        --des "20240601" \
        --lradj "HalfLR" \
        --learning_rate 0.0001 \
        --model_id "omdata4_test1" \
        --inverse
done

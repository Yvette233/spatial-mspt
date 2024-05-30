model_name="Climatology"

for i in {1..30}
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate_11/ \
        --data_path oisst_lat_14.0_lon_112.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --pred_len $i \
        --enc_in 14 \
        --dec_in 14 \
        --c_out 14 \
        --des "20240527" \
        --lradj "HalfLR" \
        --model_id "omdata1_test1" \
        --inverse
done

for i in {1..30}
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate_11/ \
        --data_path oisst_lat_14.0_lon_118.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --pred_len $i \
        --enc_in 14 \
        --dec_in 14 \
        --c_out 14 \
        --des "20240527" \
        --lradj "HalfLR" \
        --model_id "omdata2_test1" \
        --inverse
done

for i in {1..30}
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate_11/ \
        --data_path oisst_lat_20.0_lon_112.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --pred_len $i \
        --enc_in 14 \
        --dec_in 14 \
        --c_out 14 \
        --des "20240527" \
        --lradj "HalfLR" \
        --model_id "omdata3_test1" \
        --inverse
done

for i in {1..30}
do
    python -u run.py \
        --is_training 1 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate_11/ \
        --data_path oisst_lat_20.0_lon_118.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --pred_len $i \
        --enc_in 14 \
        --dec_in 14 \
        --c_out 14 \
        --des "20240527" \
        --lradj "HalfLR" \
        --model_id "omdata4_test1" \
        --inverse
done

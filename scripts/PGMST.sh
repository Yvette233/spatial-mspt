model_name="PGMST"

# for i in 10 15 20 25 30
# do
#     python -u run.py \
#         --is_training 1 \
#         --model $model_name \
#         --data custom \
#         --root_path ./dataset/multivariate/ \
#         --data_path oisst_lat_14.0_lon_112.0.csv \
#         --features MS \
#         --target sst \
#         --freq d \
#         --seq_len 365 \
#         --pred_len $i \
#         --enc_in 11 \
#         --dec_in 11 \
#         --c_out 11 \
#         --e_layers 2 \
#         --dropout 0.1 \
#         --d_model 128 \
#         --d_ff 512 \
#         --train_epochs 10 \
#         --batch_size 32 \
#         --patience 3 \
#         --des "20240701" \
#         --lradj "HalfLR" \
#         --learning_rate 0.0001 \
#         --model_id "omdata1_test1" \
#         --inverse
# done

# for i in 10 15 20 25 30
# do
#     python -u run.py \
#         --is_training 1 \
#         --model $model_name \
#         --data custom \
#         --root_path ./dataset/multivariate/ \
#         --data_path oisst_lat_14.0_lon_118.0.csv \
#         --features MS \
#         --target sst \
#         --freq d \
#         --seq_len 365 \
#         --pred_len $i \
#         --enc_in 11 \
#         --dec_in 11 \
#         --c_out 11 \
#         --e_layers 2 \
#         --dropout 0.1 \
#         --d_model 128 \
#         --d_ff 512 \
#         --train_epochs 10 \
#         --batch_size 32 \
#         --patience 3 \
#         --des "20240701" \
#         --lradj "HalfLR" \
#         --learning_rate 0.0001 \
#         --model_id "omdata2_test1" \
#         --inverse
# done

for i in 64 128 256 512
do
    python -u run.py \
        --is_training 0 \
        --model $model_name \
        --data custom \
        --root_path ./dataset/multivariate/ \
        --data_path oisst_lat_14.0_lon_112.0.csv \
        --features MS \
        --target sst \
        --freq d \
        --seq_len 365 \
        --pred_len 30 \
        --enc_in 11 \
        --dec_in 11 \
        --c_out 11 \
        --e_layers 2 \
        --dropout 0.1 \
        --d_model $i \
        --d_ff $((4 * $i)) \
        --train_epochs 10 \
        --batch_size 32 \
        --patience 3 \
        --des "20240701" \
        --lradj "HalfLR" \
        --learning_rate 0.0001 \
        --model_id "omdata1_test1" \
        --inverse \
        --individual
done

# for i in 10 15 20 25 30
# do
#     python -u run.py \
#         --is_training 1 \
#         --model $model_name \
#         --data custom \
#         --root_path ./dataset/multivariate/ \
#         --data_path oisst_lat_20.0_lon_112.0.csv \
#         --features MS \
#         --target sst \
#         --freq d \
#         --seq_len 365 \
#         --pred_len $i \
#         --enc_in 11 \
#         --dec_in 11 \
#         --c_out 11 \
#         --e_layers 2 \
#         --dropout 0.1 \
#         --d_model 128 \
#         --d_ff 512 \
#         --train_epochs 10 \
#         --batch_size 32 \
#         --patience 3 \
#         --des "20240701" \
#         --lradj "HalfLR" \
#         --learning_rate 0.0001 \
#         --model_id "omdata3_test1" \
#         --inverse
# done

# for i in 10 15 20 25 30
# do
#     python -u run.py \
#         --is_training 1 \
#         --model $model_name \
#         --data custom \
#         --root_path ./dataset/multivariate/ \
#         --data_path oisst_lat_20.0_lon_118.0.csv \
#         --features MS \
#         --target sst \
#         --freq d \
#         --seq_len 365 \
#         --pred_len $i \
#         --enc_in 11 \
#         --dec_in 11 \
#         --c_out 11 \
#         --e_layers 2 \
#         --dropout 0.1 \
#         --d_model 128 \
#         --d_ff 512 \
#         --train_epochs 10 \
#         --batch_size 32 \
#         --patience 3 \
#         --des "20240701" \
#         --lradj "HalfLR" \
#         --learning_rate 0.0001 \
#         --model_id "omdata4_test1" \
#         --inverse
# done

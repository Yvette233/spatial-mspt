model_name="TimesNet"

for layer in 1 2 3; do
    for d_model in 16 32 64; do
        for top_k in 1 2 3 4 5; do
            for pred_len in 10 15 20 25 30; do
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
                    --pred_len $pred_len \
                    --enc_in 11 \
                    --dec_in 11 \
                    --c_out 11 \
                    --e_layers $layer \
                    --top_k $top_k \
                    --dropout 0.1 \
                    --d_model $d_model \
                    --d_ff $d_model \
                    --train_epochs 10 \
                    --batch_size 32 \
                    --patience 3 \
                    --des "20240701" \
                    --lradj "HalfLR" \
                    --learning_rate 0.0001 \
                    --model_id "omdata1_test"${top_k} \
                    --inverse
            done
        done
    done
done

# for layer in 1 2 3; do
#     for d_model in 16 32 64; do
#         for top_k in 1 2 3 4 5; do
#             for pred_len in 10 15 20 25 30; do
#                 python -u run.py \
#                     --is_training 1 \
#                     --model $model_name \
#                     --data custom \
#                     --root_path ./dataset/multivariate/ \
#                     --data_path oisst_lat_14.0_lon_118.0.csv \
#                     --features MS \
#                     --target sst \
#                     --freq d \
#                     --seq_len 365 \
#                     --pred_len $pred_len \
#                     --enc_in 11 \
#                     --dec_in 11 \
#                     --c_out 11 \
#                     --e_layers $layer \
#                     --top_k $top_k \
#                     --dropout 0.1 \
#                     --d_model $d_model \
#                     --d_ff $d_model \
#                     --train_epochs 10 \
#                     --batch_size 32 \
#                     --patience 3 \
#                     --des "20240701" \
#                     --lradj "HalfLR" \
#                     --learning_rate 0.0001 \
#                     --model_id "omdata2_test"${top_k} \
#                     --inverse
#             done
#         done
#     done
# done

# for layer in 1 2 3; do
#     for d_model in 16 32 64; do
#         for top_k in 1 2 3 4 5; do
#             for pred_len in 10 15 20 25 30; do
#                 python -u run.py \
#                     --is_training 1 \
#                     --model $model_name \
#                     --data custom \
#                     --root_path ./dataset/multivariate/ \
#                     --data_path oisst_lat_20.0_lon_112.0.csv \
#                     --features MS \
#                     --target sst \
#                     --freq d \
#                     --seq_len 365 \
#                     --pred_len $pred_len \
#                     --enc_in 11 \
#                     --dec_in 11 \
#                     --c_out 11 \
#                     --e_layers $layer \
#                     --top_k $top_k \
#                     --dropout 0.1 \
#                     --d_model $d_model \
#                     --d_ff $d_model \
#                     --train_epochs 10 \
#                     --batch_size 32 \
#                     --patience 3 \
#                     --des "20240701" \
#                     --lradj "HalfLR" \
#                     --learning_rate 0.0001 \
#                     --model_id "omdata3_test"${top_k} \
#                     --inverse
#             done
#         done
#     done
# done

for layer in 1 2 3; do
    for d_model in 16 32 64; do
        for top_k in 1 2 3 4 5; do
            for pred_len in 10 15 20 25 30; do
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
                    --pred_len $pred_len \
                    --enc_in 11 \
                    --dec_in 11 \
                    --c_out 11 \
                    --e_layers $layer \
                    --top_k $top_k \
                    --dropout 0.1 \
                    --d_model $d_model \
                    --d_ff $d_model \
                    --train_epochs 10 \
                    --batch_size 32 \
                    --patience 3 \
                    --des "20240701" \
                    --lradj "HalfLR" \
                    --learning_rate 0.0001 \
                    --model_id "omdata4_test"${top_k} \
                    --inverse
            done
        done
    done
done
    
    
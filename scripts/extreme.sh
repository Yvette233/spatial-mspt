#!/bin/bash

# 定义模型名称列表
MODEL_NAMES=("MSPT_0725" )  # 将这里的模型名称替换为实际的模型名称
DATA_PATHS=("oisst_lat_14.0_lon_118.0.csv" )

# 定义固定参数
PRED_LEN=30
IS_TRAINING=1
DATA_TYPE="mhw"
ROOT_PATH="./dataset/multivariate/"
FEATURES="MS"
TARGET="sst"
FREQ="d"
SEQ_LEN=365
ENC_IN=11
DEC_IN=11
C_OUT=11
E_LAYERS=2
D_LAYERS=1
D_MODEL=128
D_FF=512
DROPOUT=0.1
TRAIN_EPOCHS=10
BATCH_SIZE=32
PATIENCE=3
DESCRIPTION="20250103"
LR_ADJ="Constant"
LEARNING_RATE=0.0001

INVERSE_FLAG="--inverse"

# 循环遍历每个模型名称并运行命令
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    for DATA_PATH in "${DATA_PATHS[@]}"
    do
        echo "========================================"
        echo "正在运行模型: $MODEL_NAME, 数据集: $DATA_PATH"
        echo "========================================"


        # 计算 label_len
        LABEL_LEN=48 # 默认值

        MODEL_ID="extreme_omdata1"
        if [ "$DATA_PATH" == "oisst_lat_14.0_lon_118.0.csv" ]; then
            MODEL_ID="extreme_omdata2"
        elif [ "$DATA_PATH" == "oisst_lat_20.0_lon_112.0.csv" ]; then
            MODEL_ID="extreme_omdata3"
        elif [ "$DATA_PATH" == "oisst_lat_20.0_lon_118.0.csv" ]; then
            MODEL_ID="extreme_omdata4"
        fi

        # 构建并运行命令
        python -u run_extreme.py \
            --is_training $IS_TRAINING \
            --model "$MODEL_NAME" \
            --data "$DATA_TYPE" \
            --root_path "$ROOT_PATH" \
            --data_path "$DATA_PATH" \
            --features "$FEATURES" \
            --target "$TARGET" \
            --freq "$FREQ" \
            --seq_len $SEQ_LEN \
            --label_len $LABEL_LEN \
            --pred_len $PRED_LEN \
            --enc_in $ENC_IN \
            --dec_in $DEC_IN \
            --c_out $C_OUT \
            --e_layers $E_LAYERS \
            --d_layers $D_LAYERS \
            --d_model $D_MODEL \
            --d_ff $D_FF \
            --dropout $DROPOUT \
            --train_epochs $TRAIN_EPOCHS \
            --batch_size $BATCH_SIZE \
            --patience $PATIENCE \
            --des "$DESCRIPTION" \
            --lradj "$LR_ADJ" \
            --learning_rate $LEARNING_RATE \
            --model_id "$MODEL_ID" \
            --top_k 4 \
            $INVERSE_FLAG
        
        # 检查命令是否成功执行
        if [ $? -ne 0 ]; then
            echo "模型 $MODEL_NAME 的数据集 $DATA_PATH 运行失败。"
            exit 1
        fi

        echo "模型 $MODEL_NAME 的数据集 $DATA_PATH 运行完成。"
    done
    echo "========================================"
    echo "模型 $MODEL_NAME 的所有数据集运行完成。"
    echo "========================================"


done
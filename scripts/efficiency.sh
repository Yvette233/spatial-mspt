#!/bin/bash

# 定义模型名称列表
MODEL_NAMES=("MSPT_0725" "TransDtSt_Part" "FC_LSTM" "iTransformer" "MICN" "PatchTST" "TimesNet" "DLinear")  # 将这里的模型名称替换为实际的模型名称


# 定义固定参数
PRED_LEN=30
IS_TRAINING=1
DATA_TYPE="custom"
ROOT_PATH="./dataset/multivariate/"
DATA_PATH="oisst_lat_14.0_lon_112.0.csv"
FEATURES="MS"
TARGET="sst"
FREQ="d"
SEQ_LEN=365
ENC_IN=11
DEC_IN=11
C_OUT=11
E_LAYERS=2
D_LAYERS=1
D_MODEL=64
DROPOUT=0.1
TRAIN_EPOCHS=10
BATCH_SIZE=32
PATIENCE=3
DESCRIPTION="20241229"
LR_ADJ="HalfLR"
LEARNING_RATE=0.0001
MODEL_ID="efficiency"
INVERSE_FLAG="--inverse"

# 循环遍历每个模型名称并运行命令
for MODEL_NAME in "${MODEL_NAMES[@]}"
do
    echo "========================================"
    echo "正在运行模型: $MODEL_NAME, pred_len: $PRED_LEN"
    echo "========================================"

    # 计算 label_len
    LABEL_LEN=48 # 默认值
    if [ "$MODEL_NAME" == "TransDtSt_Part" ]; then
        # TransDtSt_Part 模型的 label_len 为 seq_len - pred_len
        LABEL_LEN=$((SEQ_LEN - PRED_LEN))
    elif [ "$MODEL_NAME" == "MICN" ]; then
        # MICN 模型的 label_len 为 seq_len
        LABEL_LEN=$SEQ_LEN
    fi

    D_FF=$((D_MODEL * 4))
    if [ "$MODEL_NAME" == "TimesNet" ]; then
        D_FF=$D_MODEL
    fi

    # 构建并运行命令
    python -u run_efficiency.py \
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
        $INVERSE_FLAG \

    # 检查命令是否成功执行
    if [ $? -ne 0 ]; then
        echo "模型 $MODEL_NAME 运行失败。"
        exit 1
    fi

    echo "模型 $MODEL_NAME 运行完成。"
    echo
done

echo "所有模型均已运行完成。"
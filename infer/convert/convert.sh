#!/bin/bash
AIR_PATH=$1
OM_PATH=$2

echo "Input path of AIR file: ${AIR_PATH}"
echo "Output path of OM file: ${OM_PATH}"

atc --framework=1 \
    --model="${AIR_PATH}" \
    --output="${OM_PATH}" \
    --input_format=NHWC \
    --input_shape="actual_input_1:1,1024,2048,3" \
    --output_type=FP32 \
    --soc_version=Ascend310 \
    --op_select_implmode="high_precision"
exit 0

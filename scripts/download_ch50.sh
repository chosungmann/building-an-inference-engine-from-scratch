#!/bin/sh
# 50장에서 견줄 웹 엔진 셋과, 그 엔진들이 읽는 모델 파일을 받는다.
#
# 사용법: sh scripts/download_ch50.sh
#
# 엔진은 npm에서 판을 고정해 third_party/web/ 에 받는다(저장소에 넣지
# 않는다). 모델은 Hugging Face Hub에서 판을 고정해 models/ch50/ 에 받고
# SHA-256 해시를 확인한다. Llama 3.2를 뺀 나머지는 Apache-2.0이나 MIT다.
#
#   onnxruntime-web 1.31.1 (MIT)            ONNX Runtime Web
#   @huggingface/transformers 4.3.2 (Apache-2.0)   Transformers.js
#   @mlc-ai/web-llm 0.2.89 (Apache-2.0)     WebLLM
#
#   HuggingFaceTB/SmolLM2-135M-Instruct     ONNX로 내보낸 가중치(onnx/)
#   mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC  MLC가 컴파일한 가중치
#   mlc-ai/binary-mlc-llm-libs              MLC가 컴파일한 모델의 코드(.wasm)
#   onnx-community/whisper-tiny             ONNX로 내보낸 Whisper tiny
#   mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC, -q4f16_1-MLC
#                                           MLC가 컴파일한 Llama 3.2 1B
#                                           (Llama 3.2 Community License)
#
# 받는 양은 4.4GB쯤이다.
set -eu

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MODELS="${BADAK_MODELS:-$ROOT/models}/ch50"
HUB="https://huggingface.co"

# 1. 엔진.
mkdir -p "$ROOT/third_party/web"
(cd "$ROOT/third_party/web" && npm install --silent --no-audit --no-fund --ignore-scripts \
    onnxruntime-web@1.31.1 @huggingface/transformers@4.3.2 @mlc-ai/web-llm@0.2.89)

# 2. 모델. "둘 곳 주소 해시"의 줄들이다.
SMOLLM2="$HUB/HuggingFaceTB/SmolLM2-135M-Instruct/resolve/12fd25f77366fa6b3b4b768ec3050bf629380bac"
MLC="$HUB/mlc-ai/SmolLM2-135M-Instruct-q0f16-MLC/resolve/06fc409f2212adff29d93cbc742c1d6957302df5"
LIBS="https://raw.githubusercontent.com/mlc-ai/binary-mlc-llm-libs/5b7251f3bdce957be2a7bf73a6f6c560d9874df4/web-llm-models/v0_2_87/base"
WHISPER="$HUB/onnx-community/whisper-tiny/resolve/ff4177021cc41f7db950912b73ea4fdf7d01d8e7"
# Llama 3.2 1B를 MLC가 컴파일한 것 둘: 16비트 그대로와 4비트로 줄인 것.
LLAMA_F16="$HUB/mlc-ai/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/2c52165f529227534c21d83f140293d86f4eaaa3"
LLAMA_Q4="$HUB/mlc-ai/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/2a37b0a5ecb622d51ddc2fac74de0b95872affd7"
# Transformers.js는 "모델 이름/파일"의 꼴로 찾는다. WebLLM은 "모델/resolve/main/파일"로 찾는다.
ONNX="hf/HuggingFaceTB/SmolLM2-135M-Instruct"
MLC_OUT="mlc/SmolLM2-135M-Instruct-q0f16-MLC/resolve/main"
WHISPER_OUT="hf/onnx-community/whisper-tiny"

FILES="
$ONNX/config.json $SMOLLM2/config.json 8eb740e8bbe4cff95ea7b4588d17a2432deb16e8075bc5828ff7ba9be94d982a
$ONNX/generation_config.json $SMOLLM2/generation_config.json 87b916edaaab66b3899b9d0dd0752727dff6666686da0504d89ae0a6e055a013
$ONNX/tokenizer.json $SMOLLM2/tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
$ONNX/tokenizer_config.json $SMOLLM2/tokenizer_config.json 4ec77d44f62efeb38d7e044a1db318f6a939438425312dfa333b8382dbad98df
$ONNX/special_tokens_map.json $SMOLLM2/special_tokens_map.json 2b7379f3ae813529281a5c602bc5a11c1d4e0a99107aaa597fe936c1e813ca52
$ONNX/onnx/model_fp16.onnx $SMOLLM2/onnx/model_fp16.onnx e85e706ce14a3499e1f698c0d638b430df75b5f67131d1516bd0d46b844dfb2e
$ONNX/onnx/model_q4f16.onnx $SMOLLM2/onnx/model_q4f16.onnx 9358cd4ce037c304621f8c194a525607ae7c5ea73239fcae4c21bd02f2e34ff7
$ONNX/onnx/model_q4.onnx $SMOLLM2/onnx/model_q4.onnx 933577110303a2964096d19b6f15d3b4639bef7f99481ac0b61d9f3ad72f392a
$MLC_OUT/mlc-chat-config.json $MLC/mlc-chat-config.json 84fa8cda47c443589d05c7295339b0c58c15d964e7f2df53873366187789bb59
$MLC_OUT/ndarray-cache.json $MLC/ndarray-cache.json bb1307859901ebe833a1ad9f25b5a2babed8bbcfe8ea1d7e5b64db52e1c17149
$MLC_OUT/tensor-cache.json $MLC/tensor-cache.json bb1307859901ebe833a1ad9f25b5a2babed8bbcfe8ea1d7e5b64db52e1c17149
$MLC_OUT/tokenizer.json $MLC/tokenizer.json 9ca9acddb6525a194ec8ac7a87f24fbba7232a9a15ffa1af0c1224fcd888e47c
$MLC_OUT/tokenizer_config.json $MLC/tokenizer_config.json a27f638bd2831f5c3dea654a75838930f2b11fbe550c4d4e1d5d7bd07157b2ee
$MLC_OUT/params_shard_0.bin $MLC/params_shard_0.bin dbe321b43856cffd6920fc39ce1e9750556c3698aae47ca561652548d23dc2cd
$MLC_OUT/params_shard_1.bin $MLC/params_shard_1.bin 90eb8da289949ce6ea2fceec85228966fcc6b8cd962504ce0a58e21a0e308f51
$MLC_OUT/params_shard_2.bin $MLC/params_shard_2.bin ffb047fe03c824f7f32fb7abad18c6265ec31400b2b98e32d442deda8b030ce8
$MLC_OUT/params_shard_3.bin $MLC/params_shard_3.bin 8feb2e304125627dc231c30d4457fe9c5df5709b9e2858046ac3463175401ef7
$MLC_OUT/params_shard_4.bin $MLC/params_shard_4.bin 1cda5ecde438a850307b9490514e58235c64d640399fa68a4c0c28fcff84a1e4
$MLC_OUT/params_shard_5.bin $MLC/params_shard_5.bin 6df9612bec9d030861635bd33f3cbeef82eba604fed050b168252f965916de35
$MLC_OUT/params_shard_6.bin $MLC/params_shard_6.bin 3cdabc62da6bf632d0000920cf0785f93a7b38140a2ab1096f8e029dc2382ebb
$MLC_OUT/params_shard_7.bin $MLC/params_shard_7.bin ef917ee0421e0fa69bcde5bfc7ba50a96028756bb30181004d34b3babceb88fc
mlc/SmolLM2-135M-Instruct-q0f16_cs1k-webgpu.wasm $LIBS/SmolLM2-135M-Instruct-q0f16_cs1k-webgpu.wasm 853eecabfe29da72742904c345581de8afbdb2e0ebc671daa79199d8f550a3fd
$WHISPER_OUT/config.json $WHISPER/config.json 46aeea0a406afbeb563fc8e59ca10609203df4299af6a83f73752fef369efd2d
$WHISPER_OUT/generation_config.json $WHISPER/generation_config.json f5c67e5a4f7102f8cb4d058bc95da276bbc19eeec997267c3bb0f25ef68facd1
$WHISPER_OUT/preprocessor_config.json $WHISPER/preprocessor_config.json a6a76d28c93edb273669eb9e0b0636a2bddbb1272c3261e47b7ca6dfdbac1b8d
$WHISPER_OUT/tokenizer.json $WHISPER/tokenizer.json 27fc476bfe7f17299480be2273fc0608e4d5a99aba2ab5dec5374b4482d1a566
$WHISPER_OUT/tokenizer_config.json $WHISPER/tokenizer_config.json 2a4c4281cf9f51ac6ccc406fdc711a087afe6530f671fa7b80953edc498275ce
$WHISPER_OUT/special_tokens_map.json $WHISPER/special_tokens_map.json e67ae3a0aaa99abcd9f187138e12db1f65c16a14761c50ef10eef2c174a7a691
$WHISPER_OUT/added_tokens.json $WHISPER/added_tokens.json 9715fd2243b6f06a5858b5e32950d2853f73dd5bc201aafcf76f5082a2d8acd1
$WHISPER_OUT/normalizer.json $WHISPER/normalizer.json bf1c507dc8724ca9cf9903640dacfb69dae2f00edee4f21ceba106a7392f26dd
$WHISPER_OUT/vocab.json $WHISPER/vocab.json 50d6a919f0a0601d56a04eb583c780d18553aa388254ba3158eb6a00f13e2c1a
$WHISPER_OUT/merges.txt $WHISPER/merges.txt 2df2990a395e35e8dfbc7511e08c12d56018d8d04691e0133e5d63b21e154dc6
$WHISPER_OUT/onnx/encoder_model_fp16.onnx $WHISPER/onnx/encoder_model_fp16.onnx fc4bf9f3fadc450b128c3ef0711a4e61a55797447b6a700abe7fb97d9f549518
$WHISPER_OUT/onnx/decoder_model_merged_fp16.onnx $WHISPER/onnx/decoder_model_merged_fp16.onnx a1aac8aa5f4e9e835e343025edf5e4d1644ab6f629c0f8c21e121868e60b3f6c
$WHISPER_OUT/onnx/encoder_model.onnx $WHISPER/onnx/encoder_model.onnx 6642befb640f950d4a8cbbd17834d59e7e75f575b81ccf213e06b050623ab1dd
$WHISPER_OUT/onnx/decoder_model_merged.onnx $WHISPER/onnx/decoder_model_merged.onnx 8d20f4157407006e871d63ca0a3c54dddd7db33dfc4ee076960f1f8b2763ce3e
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/mlc-chat-config.json $LLAMA_F16/mlc-chat-config.json a97344cdacabf785075233186975f117473c1a027a47b6b5cfc96d2dd8f2d683
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/ndarray-cache.json $LLAMA_F16/ndarray-cache.json 3cf315345097d3a714c0349750e957fbfc15f20f0503d0f2094932ad81c22513
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/tensor-cache.json $LLAMA_F16/tensor-cache.json 3cf315345097d3a714c0349750e957fbfc15f20f0503d0f2094932ad81c22513
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/tokenizer.json $LLAMA_F16/tokenizer.json 79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/tokenizer_config.json $LLAMA_F16/tokenizer_config.json 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_0.bin $LLAMA_F16/params_shard_0.bin c214e176a8e43329defccb3a6d6e11b75a0ecff5b967a18d898849a103a574d5
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_1.bin $LLAMA_F16/params_shard_1.bin 700b152607e1b2d3e89de359b5d7ae216383c34077aa02aac6e52477f45eb75d
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_2.bin $LLAMA_F16/params_shard_2.bin 64746fcec71a162538c5ca36ee878cbdce9132687ec7afc730a0824752a02062
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_3.bin $LLAMA_F16/params_shard_3.bin 0d2d36bb5e0a47709d62d6a8320247bd69b5124ce338739a42a149dac93fac86
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_4.bin $LLAMA_F16/params_shard_4.bin 1b6c2c9adcd35c1da176e6608f4d4fd560d09b800c0f91c16b2eecb3a6fee0ee
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_5.bin $LLAMA_F16/params_shard_5.bin f0ba0a8aa0d4df0f87e3d28a6a9bd87963a951ccdce142ee7fee6606fd7a404d
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_6.bin $LLAMA_F16/params_shard_6.bin dcc01d9dfbd65023ea065cc21b9824b344e7dc55e32ba401f9e733a55034f1f0
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_7.bin $LLAMA_F16/params_shard_7.bin f09f23913b26ab368405a8069bf274db56adce3c36e330c1e930e9e421eafe47
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_8.bin $LLAMA_F16/params_shard_8.bin c353c67b5f35e707862747d69cae85bd6b292caa324bf2ef2e71614f97047c1f
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_9.bin $LLAMA_F16/params_shard_9.bin afb1f65d970838b4fb5f9fa4ab73e345be9a10c3eb4135e320402a2e197f92e1
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_10.bin $LLAMA_F16/params_shard_10.bin 1089e327f40ec0e5180eb2c85d129bcc4f4708cde820b017fb16eca815d9a363
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_11.bin $LLAMA_F16/params_shard_11.bin bd25f6fa58c91930df8604e55e08b8875a288a7448a2448c9856da50314bec49
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_12.bin $LLAMA_F16/params_shard_12.bin 6d1a815cea5efd5887c5f2a170c4863d5b4dda283ffda6e977cf559856a0249e
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_13.bin $LLAMA_F16/params_shard_13.bin 159d759832bef0f532ffa90d02c7ebe69cc5c57956c2f0abbe87d816e7715396
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_14.bin $LLAMA_F16/params_shard_14.bin 16475c045a3377a03403e2df89db78f48197961771aac97ea7aeeddae9a82235
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_15.bin $LLAMA_F16/params_shard_15.bin 9e6bc762566ab8d9af1226ba8ea958d1d9a1270dfd31058e0eee74ca5922b107
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_16.bin $LLAMA_F16/params_shard_16.bin b6fecae8ad9b966d8aa71f81bb21eaf5c566a1c39b8f095778e7058011e8cf0d
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_17.bin $LLAMA_F16/params_shard_17.bin 0755b1734fb7ab64efbf03cda3c4876cff5d1bba3820362a98d51740a4b6d6d0
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_18.bin $LLAMA_F16/params_shard_18.bin 04a6bd1aede5799f69604778ad73e5ee87c25df19665edf32ba8234fa44e1484
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_19.bin $LLAMA_F16/params_shard_19.bin cb34b1a11a5d6e85bcbf6e496ab2ea4308e2b75fcd36738babd15014e89c1a0d
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_20.bin $LLAMA_F16/params_shard_20.bin 2474097267b9a9104fd244ef80753f00c0421a568c88d1a47997b1eea1ffc728
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_21.bin $LLAMA_F16/params_shard_21.bin cf85e74ac35395aea0040a4a1e2b8f92633b96ed67e9883b8685e83029459e9f
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_22.bin $LLAMA_F16/params_shard_22.bin 3d02a714b039cc4a788bad50c2d37a53286f00f36368b46e8cbe048d0e47872e
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_23.bin $LLAMA_F16/params_shard_23.bin 9af27a69d587eee0d68b37fd52fe168f4753057a1cc81f28f6e202d3ef3c0fa7
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_24.bin $LLAMA_F16/params_shard_24.bin 77be486c25e90fde59b9bbeb9650ea377d0d236f1c2ea28eaebb7290e39ad63e
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_25.bin $LLAMA_F16/params_shard_25.bin 3f9e9435ac65354fb971f79effef06b83111a727ff2b20bdbada4fca1133890c
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_26.bin $LLAMA_F16/params_shard_26.bin 20d3957663bed4d3d811a165f06a9cd6693165acf1c364f5bf275999ddf4c980
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_27.bin $LLAMA_F16/params_shard_27.bin fd3568b232db3053c9ace9860617f42633545497118172f37d4532d0ec8bf321
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_28.bin $LLAMA_F16/params_shard_28.bin 0e64dec47aaf9e16b0d178082ade7bd81dfc257cbc2bca6f1c5d615da85adfc6
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_29.bin $LLAMA_F16/params_shard_29.bin 683daa0c372a06e55d096ecf51deafd9de9c43f998a257cd728e5ee6eef5c935
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_30.bin $LLAMA_F16/params_shard_30.bin 1a73972b0c1a9eef072850f67b60e5b2c9a545c2a7fa9cccc68549f088be5c8f
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_31.bin $LLAMA_F16/params_shard_31.bin fc6ca0b3b36e741a9460fca85ef4bcdc7503af2f2bd19ef8b850ef6414b34325
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_32.bin $LLAMA_F16/params_shard_32.bin b5327e3eb2b35681320139878ef598c22ac3dda0472a0dd895c6070495a00311
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_33.bin $LLAMA_F16/params_shard_33.bin 91fa7418b540dfff9ecb0e36ac5bdad9fd82ebd3fcac9daca22ea50441b9d1bc
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_34.bin $LLAMA_F16/params_shard_34.bin e63b76285dfa11f774a881100afced1e3bb073336ab281b89ff0de75cdda8bf8
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_35.bin $LLAMA_F16/params_shard_35.bin e6e0b886106b378bee9b8f8444e62da5e13740a12feb07e351c570fb689089ac
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_36.bin $LLAMA_F16/params_shard_36.bin 897a4bd69caefa8d7ee3d1df58ef9f93d1e795ebde121c21b5216a6d145879ec
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_37.bin $LLAMA_F16/params_shard_37.bin 9d02eb690bc1987a1f019f5d898cb1fe7c7a793bf50cd60381edad6e09e08c07
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_38.bin $LLAMA_F16/params_shard_38.bin 8741a53b21b0a364f1ce78c47a1a870dbc183f54e32e47924f60b373ec0c1b82
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_39.bin $LLAMA_F16/params_shard_39.bin 0b8606795181efba208881c2f0b63de512d6bd2ed6eeac72adba8c12aeab4d83
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_40.bin $LLAMA_F16/params_shard_40.bin dc64681d16432fd3b5e7bdb5d5019c178511bd998a01d9f394d64dc121a618ee
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_41.bin $LLAMA_F16/params_shard_41.bin 56e4a9e65e2c9fec2078d28cacedefeccfbec48828248931cee9058e7c0a03b2
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_42.bin $LLAMA_F16/params_shard_42.bin b76e03985cd9ded47cadf01f896da84e5c3bc4f8896390f91c73d5a53aab6482
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_43.bin $LLAMA_F16/params_shard_43.bin 9e043c0a7f0e03d813a87d8f721cedc60174b5926c4d7eafe8644ade3d7d17f2
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_44.bin $LLAMA_F16/params_shard_44.bin 7240a872629d1c036f03cdad66a2d7f0257139d375cdb810b27167e53669764b
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_45.bin $LLAMA_F16/params_shard_45.bin 60898c873aa6dbbfaba9ca6337001665365091484147a44c20828e6ce1edc583
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_46.bin $LLAMA_F16/params_shard_46.bin 25475ec182dd2cc867e15d5798285cd2f7fd5717dba03ecd574be0a6359c03d1
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_47.bin $LLAMA_F16/params_shard_47.bin 18ce44b582b6f2dfa109e1936417e985b01acd149f2dd93f6a39e68872005827
mlc/Llama-3.2-1B-Instruct-q0f16-MLC/resolve/main/params_shard_48.bin $LLAMA_F16/params_shard_48.bin 4408107b52beffcdce5cf0c7d4e06343e5d374afebbdf0135f6841008bf1111f
mlc/Llama-3.2-1B-Instruct-q0f16_cs1k-webgpu.wasm $LIBS/Llama-3.2-1B-Instruct-q0f16_cs1k-webgpu.wasm 9610b145e7eebea403f2fcca61e1f42e9a327c70d9a85e6c6d9edf328a6f3791
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/mlc-chat-config.json $LLAMA_Q4/mlc-chat-config.json 0ec513b54b419ad471006430686bdcffaae7102b41f7b0246fbfcde2517acc7f
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/ndarray-cache.json $LLAMA_Q4/ndarray-cache.json cf7c1825a9e0d7111bca7f65d8d9de7789f9121579e9500150469543a2975c17
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/tensor-cache.json $LLAMA_Q4/tensor-cache.json cf7c1825a9e0d7111bca7f65d8d9de7789f9121579e9500150469543a2975c17
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/tokenizer.json $LLAMA_Q4/tokenizer.json 79e3e522635f3171300913bb421464a87de6222182a0570b9b2ccba2a964b2b4
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/tokenizer_config.json $LLAMA_Q4/tokenizer_config.json 9823dcfdc1121869029da45192238e85cf44f0b232a6d9dc20e4fe6f4242a14e
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_0.bin $LLAMA_Q4/params_shard_0.bin 17f0334f88adb9a6a23cc702b1fd895e6a159d164eae2520358c8c1d3fccc9ed
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_1.bin $LLAMA_Q4/params_shard_1.bin 76b9e11e8fdb82d6d3edbba334bbf59f6fd080e8138baf6bca956af9c886c956
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_2.bin $LLAMA_Q4/params_shard_2.bin 203ba6b3283b604a8200a1f06051f85c5a9003a606ba89c2f5d47ce9d08ee2fa
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_3.bin $LLAMA_Q4/params_shard_3.bin e1bc66ba5102f6971c0ff430234405a3a5831f13ff88301f02094fe50cb0764a
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_4.bin $LLAMA_Q4/params_shard_4.bin 19d1b44307b23060f5f3acee5a0b5abf22fcfea4a3d2e8f4469d00aeec5a4ce3
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_5.bin $LLAMA_Q4/params_shard_5.bin aa2a7f2ec3a7f3ac0c42e698bec4a3ce30e8ebfabe1f9d7f4bede1ddbb92e1e8
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_6.bin $LLAMA_Q4/params_shard_6.bin 2a2d03e4a918dbd3ac1899fb155147013959edbd5cf233947d6900936fc7a419
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_7.bin $LLAMA_Q4/params_shard_7.bin 0918287f31f10dd760ed405ca10571dce9fcb621f42978cd3b5c0d2dce6b4304
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_8.bin $LLAMA_Q4/params_shard_8.bin a34bdd87a4d0ccb5baec1e5bfbd66bb407d04bb3d8d966d53c8bff9ebcae5093
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_9.bin $LLAMA_Q4/params_shard_9.bin a9b2a284c794fc23919a74210ec5cda44f9068126e37ad33c7547c812954cd01
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_10.bin $LLAMA_Q4/params_shard_10.bin bf02d53bd89d7c7f46b2f64d9d851cdf1b4a4dd1b873369e633e97056307f8d5
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_11.bin $LLAMA_Q4/params_shard_11.bin 578a5f48dac3bc367c8ad2b3da30148686366f1b5d626de2e4083a115a985e67
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_12.bin $LLAMA_Q4/params_shard_12.bin 1237ea88c9640ad3498cb59222c2a3a1d5ec47fa91a52c409f1ad9b53a4ada59
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_13.bin $LLAMA_Q4/params_shard_13.bin 42e95969cf75ffce16661b77704cdb556dcc499070b026ce8f350235348e5584
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_14.bin $LLAMA_Q4/params_shard_14.bin d7c96577cee90e7254a18af800480c249e0860b3b887e0497647032ea4eb1c59
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_15.bin $LLAMA_Q4/params_shard_15.bin 88cf9a9d4a903b765e17528be0462be1abd6b328de02b830b8d37e32deb55fbc
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_16.bin $LLAMA_Q4/params_shard_16.bin aac1472bd13261a475c06e3aaa5fc87dd78b5a2c97a2cdac7ace244158e5035e
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_17.bin $LLAMA_Q4/params_shard_17.bin e8b201b70ba99337c84fe9611e18f1e48564a273168cf5cdb6fcb173dfec749c
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_18.bin $LLAMA_Q4/params_shard_18.bin ed30fdf696b521b2ed18d91933d965f9aaecf23601d1f5b0a215598cc2bfae69
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_19.bin $LLAMA_Q4/params_shard_19.bin d93f253fdf638757eb0003d7d9b0f75ae7c9c4e36323c5df5f486dcde9790249
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_20.bin $LLAMA_Q4/params_shard_20.bin 2bcd8a5d7a3841cc9ded6b3d46fe63adb845dbabf83ee08164b8676161e8cd6a
mlc/Llama-3.2-1B-Instruct-q4f16_1-MLC/resolve/main/params_shard_21.bin $LLAMA_Q4/params_shard_21.bin 2b0072e9cbdc7698cf2eefe9355e2281b1a817f58894f1b3b2f15e103de799c3
mlc/Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm $LIBS/Llama-3.2-1B-Instruct-q4f16_1_cs1k-webgpu.wasm 8dbd7e81d39291f2666176b846f36d924a603e9480bac739ee5fc0d22f74b1fd
"

echo "$FILES" | while read -r out url expected; do
    [ -n "$out" ] || continue
    mkdir -p "$(dirname "$MODELS/$out")"
    if [ ! -f "$MODELS/$out" ]; then
        echo "내려받는 중: $out"
        curl -fsSL "$url" -o "$MODELS/$out.partial"
        mv "$MODELS/$out.partial" "$MODELS/$out"
    fi
    actual=$(shasum -a 256 "$MODELS/$out" | cut -d ' ' -f 1)
    if [ "$actual" != "$expected" ]; then
        echo "해시가 다릅니다: $out ($actual)" >&2
        exit 1
    fi
done
echo "준비되었습니다: $MODELS, $ROOT/third_party/web"

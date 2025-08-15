python \
-m vllm.entrypoints.openai.api_server \
--host 0.0.0.0 \
--port 8999 \
--model /agent/share/public/common-models/Qwen-Qwen2.5-VL-72B-Instruct/ \
--served-model-name Qwen2.5VL-72B \
--max-model-len=16384 \
--tensor_parallel_size 2 \
--limit-mm-per-prompt image=30
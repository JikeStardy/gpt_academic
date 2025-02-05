from typing import List, Dict
from .bridge_all import show_models, oai_std_model_name_mappings, tokenizer_gpt35, get_token_num_gpt35
from toolbox import get_conf, trimmed_format_exc, read_model_name_and_settings
from .oai_std_model_template import get_predict_function
from loguru import logger

API_URL_REDIRECT = get_conf("API_URL_REDIRECT")

siliconflow_endpoint = "https://api.siliconflow.cn/v1/chat/completions"
if siliconflow_endpoint in API_URL_REDIRECT: 
    siliconflow_endpoint = API_URL_REDIRECT[siliconflow_endpoint]

SILICONFLOW_PREFIX = "siliconflow-"

def register_siliconflow_model(avail_models: List[str], model_info: Dict):
    for model in [m for m in avail_models if m.startswith(SILICONFLOW_PREFIX)]:
        # 为接入siliconflow平台，设计了此接口，例子：avail_models = ["siliconflow-mixtral-8x7b(max_token=6666)"]
        # 其中
        #   "siliconflow-"      是前缀（必要）
        #   "mixtral-8x7b"      是模型名（必要）
        #   "(max_token=6666)"  是配置（非必要）
        
        try:
            # 加载模型名称
            real_model_name, model_settings = read_model_name_and_settings(model)
            show_models[model] = real_model_name
            oai_std_model_name_mappings[real_model_name] = real_model_name.replace(SILICONFLOW_PREFIX, "")
        
            # 加载模型
            try:
                siliconflow_noui, siliconflow_ui = get_predict_function(
                    api_key_conf_name="SILICONFLOW_API_KEY", max_output_token=model_settings.get('max_output_token', 4096), disable_proxy=False
                )
                model_info.update({
                    real_model_name:{
                        "fn_with_ui": siliconflow_ui,
                        "fn_without_ui": siliconflow_noui,
                        "endpoint": siliconflow_endpoint,
                        "can_multi_thread": True,
                        "max_token": model_settings.get('max_token', 4096),
                        "tokenizer": tokenizer_gpt35,
                        "token_cnt": get_token_num_gpt35,
                    },
                })
            except:
                logger.error(trimmed_format_exc())
        except Exception as e:
            logger.error(f"siliconflow模型 {model} 的配置有误，请检查配置文件: {e}")
            continue
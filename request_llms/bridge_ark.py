from typing import List, Dict
from .bridge_all import show_models, oai_std_model_name_mappings, tokenizer_gpt35, get_token_num_gpt35
from toolbox import get_conf, trimmed_format_exc, read_model_name_and_settings
from .oai_std_model_template import get_predict_function
from loguru import logger

API_URL_REDIRECT = get_conf("API_URL_REDIRECT")

ark_endpoint = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
if ark_endpoint in API_URL_REDIRECT: 
    ark_endpoint = API_URL_REDIRECT[ark_endpoint]

ARK_PREFIX = "ark-"

def register_ark_model(avail_models: List[str], model_info: Dict):
    for model in [m for m in avail_models if m.startswith(ARK_PREFIX)]:
        # 为接入ark平台，设计了此接口，例子：avail_models = ["ark-mixtral-8x7b(model_id=ep-123-zzzz)"]
        # 其中
        #   "ark-"                    是前缀（必要）
        #   "mixtral-8x7b"            是模型名（必要）
        #   "(model_id=ep-123-zzzz)"  是配置（必要）
        
        try:
            # 加载模型名称
            real_model_name, model_settings = read_model_name_and_settings(model, prefix=ARK_PREFIX)
            show_models[model] = real_model_name
            oai_std_model_name_mappings[real_model_name] = model_settings['model_id']
        
            # 加载模型
            try:
                ark_noui, ark_ui = get_predict_function(
                    api_key_conf_name="ARK_API_KEY", max_output_token=model_settings.get('max_output_token', 4096), disable_proxy=False
                )
                model_info.update({
                    model:{
                        "fn_with_ui": ark_ui,
                        "fn_without_ui": ark_noui,
                        "endpoint": ark_endpoint,
                        "can_multi_thread": True,
                        "max_token": model_settings.get('max_token', 4096),
                        "tokenizer": tokenizer_gpt35,
                        "token_cnt": get_token_num_gpt35,
                    },
                })
            except:
                logger.error(trimmed_format_exc())
        except:
            logger.error(f"ark模型 {model} 的配置有误，请检查配置文件。")
            continue
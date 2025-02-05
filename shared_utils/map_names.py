import re
mapping_dic = {
    # "qianfan": "qianfan（文心一言大模型）",
    # "zhipuai": "zhipuai（智谱GLM4超级模型🔥）",
    # "gpt-4-1106-preview": "gpt-4-1106-preview（新调优版本GPT-4🔥）",
    # "gpt-4-vision-preview": "gpt-4-vision-preview（识图模型GPT-4V）",
}

rev_mapping_dic = {}
for k, v in mapping_dic.items():
    rev_mapping_dic[v] = k

def map_model_to_friendly_names(m):
    if m in mapping_dic:
        return mapping_dic[m]
    return m

def map_friendly_names_to_model(m):
    if m in rev_mapping_dic:
        return rev_mapping_dic[m]
    return m

def read_one_api_model_name(model: str):
    """return real model name and max_token from model_cfg_name like `xxx-gpt-3.5(max_token=...)`.
    """
    max_token_pattern = r"\(max_token=(\d+)\)"
    match = re.search(max_token_pattern, model)
    if match:
        max_token_tmp = match.group(1)  # 获取 max_token 的值
        max_token_tmp = int(max_token_tmp)
        model = re.sub(max_token_pattern, "", model)  # 从原字符串中删除 "(max_token=...)"
    else:
        max_token_tmp = 4096
    return model, max_token_tmp

def read_model_name_and_settings(model: str):
    settings_pattern = r"\((.*?)\)"
    match = re.search(settings_pattern, model)
    settings = {}
    if match:
        settings_str = match.group(1)
        settings_list = settings_str.split(',')
        for setting in settings_list:
            if '=' not in setting:
                raise ValueError(f"Invalid setting format: {setting}")
            key, value = setting.split('=', 1)
            value = value.strip()
            try:
                # 尝试转换为整数
                parsed_value = int(value)
            except ValueError:
                try:
                    # 转换为浮点数
                    parsed_value = float(value)
                except ValueError:
                    parsed_value = value
            settings[key.strip()] = parsed_value
        origin_model_name = re.sub(settings_pattern, "", model).strip()
    return origin_model_name, settings
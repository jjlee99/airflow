from airflow.decorators import task
from pathlib import Path
import shutil, os, json
from airflow.models import Variable

TEMP_FOLDER = Variable.get("TEMP_FOLDER", default_var="/opt/airflow/data/temp")
RESULT_FOLDER = Variable.get("RESULT_FOLDER", default_var="/opt/airflow/data/result")
def read_config(config_path:str=None):
    if not(config_path):
        config_path = (Path(__file__).parent / "test_config.json").absolute()
        print("config_path",config_path)
    with open(config_path, "r") as f:
        config = json.load(f)
    return config

def _get_deep_info(data, *keys):
    for key in keys:
        if isinstance(data, dict) and key in data:
            data = data[key]
        else:
            return None
    return data


def get_step_info_list(*keys):
    class_map = {
        "a_class":{
            "class_id":1111,
            "classify":{
                "classify_id":1111,
                "classify_ai":{
                    "ai_id":15,
                    "ai_dir":"/opt/airflow/data/class/a_class/classify/model",
                    "processor_name":"SCUT-DLVCLab/lilt-roberta-en-base",
                    "model_name":"SCUT-DLVCLab/lilt-roberta-en-base",
                },
                "img_preprocess":{
                    "name":"a_class classify img_preprc",
                    "type":"step_list",
                    "step_list":[
                        {"name":"cache","param":{"key":"origin"}},
                        {"name":"calc_angle_set1","param":{"key":"angle1","iterations":4,"iter_save":False}},
                        {"name":"text_orientation_set","param":{"key":"angle2","iterations":4,"iter_save":False}},
                        {"name":"load","param":{"key":"origin"}},
                        {"name":"rotate","param":{"key":"angle1"}},
                        {"name":"rotate","param":{"key":"angle2"}},
                    ], 
                }
            },
            "area_cut":{},
            "ocr":{},
            "save":{
                "save_list":[
                    "classify_preprocess"
                ],
            },
        },
        "b_class":{}
    }
    return _get_deep_info(class_map,*keys)
    

def file_copy(src_file: str, dest_file: str) -> str:
    """
    파일을 복사하는 함수
    
    :param src_file: 복사할 파일 경로(문자열)
    :param dest_file: 붙여넣을 파일 경로(문자열)
    :return: 실제로 복사된 파일 경로(문자열)
    """
    print("file_copy:",dest_file)
    src = Path(src_file)
    dest = Path(dest_file)

    # 붙여넣을 폴더가 없으면 생성
    dest.parent.mkdir(parents=True, exist_ok=True)

    # 파일명과 확장자 분리
    stem = dest.stem
    suffix = dest.suffix
    count = 1

    # 파일이 이미 존재하면 숫자를 붙여서 복사
    while dest.exists():
        dest = dest.parent / f"{stem}({count}){suffix}"
        count += 1

    # 파일 복사
    shutil.copy2(src, dest)

    return str(dest)

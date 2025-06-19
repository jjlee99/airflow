from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.decorators import task, task_group
from datetime import datetime
from pathlib import Path
import os
import json
from airflow.models import Variable,XCom
# utils 모듈 임포트 (PYTHONPATH에 proj
# ect_root가 잡혀있다고 가정)
# 만약 utils 모듈이 DAG 파일과 같은 디렉토리 내에 있다면, 상대 경로 임포트를 고려하거나
# Airflow DAGs 폴더 구조에 맞게 배치해야 합니다.
# 예: dags/your_dag_file.py, dags/utils/file_util.py
from utils import file_util
from tasks.file_task import check_file_exists_branch,get_file_info_list_task,end_task,copy_results_folder_task, clear_temp_folder_task
from tasks.init_task import init_task
from tasks.img_preprocess_task import img_preprocess_task

TEMP_FOLDER = Variable.get("TEMP_FOLDER", default_var="/opt/airflow/data/temp")
RESULT_FOLDER = Variable.get("RESULT_FOLDER", default_var="/opt/airflow/data/upload")
UPLOAD_FOLDER = Variable.get("UPLOAD_FOLDER", default_var="/opt/airflow/data/upload")
def read_config_task(config_path: str = None):
    if not config_path:
        # DAGs 폴더 내에서 config 파일을 찾도록 경로 조정
        # Airflow 컨테이너 내부의 /opt/airflow/dags 경로를 기준으로 경로를 계산합니다.
        # DAG 파일이 dags/my_dag.py 라면, config 파일은 dags/test_config.json 에 있어야 합니다.
        config_path = (Path(__file__).parent / "test_config.json").absolute()
        print(f"config_path: {config_path}")
    with open(config_path, "r") as f:
        config = json.load(f)
    return config
    

# DAG 정의 (DAG 클래스 직접 사용)
with DAG(
    dag_id="image_preprocess_v1", # 이전 DAG ID와 충돌 방지를 위해 변경
    start_date=datetime(2024, 1, 1),
    schedule=None, # None으로 설정하면 수동 트리거만 가능
    catchup=False,
    tags=['image', 'batch']
) as dag:
    # 태스크 인스턴스 생성
    #여러 파일을 한 step으로 돌림
    # config = read_config_task("/opt/airflow/dags/test_config.json")
    # step_info = [
    #     {"name":"gray2","param":{}},
    #     #{"name":"denoising","param":{}},
    #     {"name":"binary1","param":{}},
    # ]
    # files_path_task = get_file_list_task(folder_path=config['target_dir'])
    # preprocess_group_partial = preprocess_group.partial(config=config,step_info=step_info)
    # taskgroup = preprocess_group_partial.expand(file_path=files_path_task)

    #A클래스 분류를 위한 전처리 스텝 목록
    test_init_task = init_task()
    
    a_class_classify_preprocess_info = file_util.get_step_info_list("a_class","classify","img_preprocess")
    check_file_branch = check_file_exists_branch(UPLOAD_FOLDER)
    file_info_list_task = get_file_info_list_task(UPLOAD_FOLDER)

    no_file_task = end_task("폴더 안에 파일이 존재하지 않습니다.")
    classify_preprocess_partial_task = img_preprocess_task.partial(step_info=a_class_classify_preprocess_info)
    classify_preprocess_task = classify_preprocess_partial_task.expand(file_info=file_info_list_task)
    classify_preprocess_result_task = copy_results_folder_task(classify_preprocess_task, last_folder="a_class")
    #a_class_classify_result_task = class_classify_result_task(classify_preprocess_result_task,"a_class")
    
    all_clear_temp_folder_task = clear_temp_folder_task()
    # 태스크 간 의존성 설정
    # XCom을 통해 데이터가 전달되므로, 태스크 실행 순서만 정의합니다.
    test_init_task>> check_file_branch
    check_file_branch >> no_file_task
    check_file_branch >> file_info_list_task >> classify_preprocess_task >> classify_preprocess_result_task
    #classify_preprocess_result_task >> all_clear_temp_folder_task

# Airflow 2.x에서 Python 스크립트 직접 실행 시에는 DAG가 파싱만 됩니다.
# 실제 테스트는 Airflow CLI를 통해 실행해야 합니다.
if __name__ == "__main__":
    print("이 스크립트를 직접 실행하면 DAG가 Airflow에 로드됩니다.")
    print("DAG를 테스트하려면 Airflow CLI 명령을 사용하세요.")
    print("예: airflow dags test image_processing_per_file_v2 2024-01-01")
    # 주의: 이 부분은 Airflow 2.x에서는 dag_instance를 직접 호출하여 테스트하는 방식이 아닙니다.
    # DAG 파일은 단순히 Airflow 스케줄러/웹서버가 읽고 파싱할 수 있도록 존재합니다.
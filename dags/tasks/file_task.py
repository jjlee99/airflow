from airflow.decorators import task
from pathlib import Path
import shutil, os
from airflow.models import Variable
import uuid

from utils.db import maria_util

TEMP_FOLDER = Variable.get("TEMP_FOLDER", default_var="/opt/airflow/data/temp")
RESULT_FOLDER = Variable.get("RESULT_FOLDER", default_var="/opt/airflow/data/result")
    

@task
def get_file_list_task(folder_path):
    files = [str(f) for f in Path(folder_path).iterdir() if f.is_file()]
    if not files:
        raise ValueError(f"No files found in: {folder_path}")
    return files

#폴더 안에 파일이 있는지 분기 있으면 get_file_info_list_task, 없으면 no_file_task
@task.branch
def check_file_exists_branch(folder_path):
    p = Path(folder_path)
    for f in p.rglob("*"):
        if f.is_file():
            return "get_file_info_list_task"
    return "end_task"

@task
def get_file_info_list_task(folder_path,**context):
    p = Path(folder_path)
    files = [str(f) for f in p.rglob("*") if f.is_file()]
    info_map_list = []
    for path in files:
        id = str(uuid.uuid4())
        info_map_list.append({"file_id":id, "file_path":path})
    return info_map_list

@task
def end_task(msg="dag을 종료합니다."):
    print(msg)
    return

@task
def copy_results_folder_task(file_infoes: list, dest_folder:str=None, last_folder:str=None, result_key:str="result", **context):
    """
    result_map["result_file_map"]에 저장된 경로의 파일들을
    지정된 폴더로 복사. 파일명 앞에 result_map["process_id"]를 붙임.
    """
    if not dest_folder:
        dest_folder = str(Path(TEMP_FOLDER) / context['run_id'])
    if last_folder:
        dest_folder = f"{dest_folder}/{last_folder}"
    os.makedirs(dest_folder, exist_ok=True)
    total_copied = 0
    print(len(file_infoes))
    for file_info in file_infoes:
        result_map = file_info[result_key]
        process_id = result_map["process_id"]
        result_file_map = result_map["result_file_map"]
        print(len(result_file_map))
        
        for key, file_path in result_file_map.items():
            if not file_path:
                print(f"경고: path가 없는 파일 정보가 있습니다. {file_info}")
                continue

            # 원본 파일명 추출
            src_filename = Path(file_path).name
            # 새 파일명 생성 (process_id + 원본 파일명)
            dest_filename = f"{process_id}_{src_filename}"
            dest_path = os.path.join(dest_folder, dest_filename)

            try:
                shutil.copy2(file_path, dest_path)
                total_copied += 1
                print(f"복사 완료: {file_path} → {dest_path}")
            except Exception as e:
                print(f"복사 실패: {file_path} → {dest_path}, 오류: {str(e)}")

    return dest_folder

@task
def clear_temp_folder_task(**context):
    run_id = context['run_id']
    folder = Path(TEMP_FOLDER) / run_id
    if folder.exists() and folder.is_dir():
        for item in folder.iterdir():
            if item.is_file():
                os.remove(item)
            elif item.is_dir():
                shutil.rmtree(item)
    return f"Cleared folder: {folder}"

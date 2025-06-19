from airflow.decorators import task
from pathlib import Path
import shutil, os
from airflow.models import Variable
import uuid

from utils.db import maria_util

TEMP_FOLDER = Variable.get("TEMP_FOLDER", default_var="/opt/airflow/data/temp")
RESULT_FOLDER = Variable.get("RESULT_FOLDER", default_var="/opt/airflow/data/result")

@task
def init_task(**context):
    dag_id = context['dag'].dag_id
    run_id = context['dag_run'].run_id
    maria_util.call_set("insertRun",params=(dag_id,run_id))
    folder = Path(TEMP_FOLDER) / run_id
    folder.mkdir(parents=True, exist_ok=True)

from airflow import DAG
from airflow.decorators import task, task_group
from datetime import datetime
from utils import file_util
import os
import random
import shutil
from tasks.file_task import get_file_info_list_task, copy_results_folder_task
from tasks.init_task import init_task
from tasks.img_preprocess_task import img_preprocess_task

# 경로 설정 (DAG 파라미터로 받거나 환경변수로 설정 가능)
DATA_DIR = "/opt/airflow/data"   # 루트
ORIGIN_IMAGE_DIR = f"{DATA_DIR}/class/a_class/classify/origin"   # 원본 문서 이미지
ORIGIN_TRUE_IMAGE_DIR = f"{ORIGIN_IMAGE_DIR}/true"   # 특정 서식 원본 문서 이미지
ORIGIN_FALSE_IMAGE_DIR = f"{ORIGIN_IMAGE_DIR}/false" # 일반 문서 이미지
READY_IMAGE_DIR = f"{DATA_DIR}/class/a_class/classify/ready"   # 증강 문서 이미지
READY_TRUE_IMAGE_DIR = f"{READY_IMAGE_DIR}/true"   # 특정 서식 증강된 문서 이미지
READY_FALSE_IMAGE_DIR = f"{READY_IMAGE_DIR}/false" # 일반 증강된 문서 이미지
PREPRC_IMAGE_DIR = f"{DATA_DIR}/class/a_class/classify/preprc"   # 전처리된 문서 이미지
PREPRC_TRUE_IMAGE_DIR = f"{PREPRC_IMAGE_DIR}/true"   # 특정 서식 전처리된 문서 이미지
PREPRC_FALSE_IMAGE_DIR = f"{PREPRC_IMAGE_DIR}/false" # 일반 전처리된 문서 이미지
NONE_DOC_IMAGE_DIR = f"{DATA_DIR}/none_class" # 비서식 일반 문서 이미지
OUTPUT_MODEL_DIR = f"{DATA_DIR}/class/a_class/classify/model"   #ai 모델

def get_image_paths_recursive(directory: str):
    """하위 폴더까지 포함하여 이미지 경로 리스트 반환"""
    image_paths = []
    for root, dirs, files in os.walk(directory):
        for f in files:
            if f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def get_image_paths(directory: str):
    """이미지 경로 리스트 반환"""
    if not os.path.exists(directory):
        return []
    return [os.path.join(directory, f) for f in os.listdir(directory) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

def get_image_count(root_dir):
    """각 디렉토리의 이미지 개수를 계산"""
    true_folder = f"{root_dir}/true"
    false_folder = f"{root_dir}/false"
    true_count = len(get_image_paths(true_folder))
    false_count = len(get_image_paths(false_folder))

    none_doc_count = len(get_image_paths_recursive(NONE_DOC_IMAGE_DIR))
    
    print(f"{root_dir} TRUE 파일 개수: {true_count}")
    print(f"{root_dir} FALSE 파일 개수: {false_count}")
    print(f"NONE_DOC_IMAGE_DIR 파일 개수: {none_doc_count}")
    
    return {
        "true_count": true_count,
        "false_count": false_count,
        "none_doc_count": none_doc_count
    }

# branch와 task 그룹
@task.branch
def check_origin_file_cnt_branch():
    true_cnt = len(get_image_paths(ORIGIN_TRUE_IMAGE_DIR))
    AUG_THRESHOLD = 200  # 기준 수치
    if true_cnt < AUG_THRESHOLD:
        return "augment_task"
    return "not_augment_task"

    
@task
def balance_false_images(root_path:str):
    counts_info = get_image_count(root_path)

    false_folder = f"{root_path}/false"
    true_count = counts_info["true_count"]
    false_count = counts_info["false_count"]
    
    needed_files = true_count - false_count
    if needed_files <= 0:
        print(f"FALSE_IMAGE_DIR에 충분한 파일이 있습니다. (필요: {needed_files})")
        return {"copied_count": 0, "total_false_count": false_count}

    # 수정 사항 1: get_image_paths_recursive만 사용
    none_doc_paths = get_image_paths_recursive(NONE_DOC_IMAGE_DIR)
    none_doc_count = len(none_doc_paths)

    if needed_files > none_doc_count:
        print(f"경고: 사용 가능 파일 부족 (요청: {needed_files}, 실제: {none_doc_count})")
        needed_files = none_doc_count
        if false_count == 0 and none_doc_count == 0:
            raise ValueError("학습을 위한 최소 데이터가 없습니다")

    # 수정 사항 3: needed_files가 0인 경우 처리
    selected_files = random.sample(none_doc_paths, needed_files) if needed_files > 0 else []
    
    os.makedirs(false_folder, exist_ok=True)
    copied_count = 0
    for src_path in selected_files:
        try:
            filename = os.path.basename(src_path)
            new_filename = f"copied_{copied_count}_{filename}"
            dst_path = os.path.join(false_folder, new_filename)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
            print(f"복사 완료: {src_path} -> {dst_path}")
        except Exception as e:
            print(f"복사 실패: {src_path} - {str(e)}")
    
    return {"copied_count": copied_count, "total_false_count": false_count + copied_count}

@task
def build_balanced_dataset(root_path):
    """균형이 맞춰진 데이터셋 구성"""
    true_folder = f"{root_path}/true"
    false_folder = f"{root_path}/false"
    true_image_paths = get_image_paths(true_folder)
    false_image_paths = get_image_paths(false_folder)
    
    dataset = []
    
    # True 라벨 데이터 추가
    for path in true_image_paths:
        dataset.append({"image_path": path, "label": 1})
    
    # False 라벨 데이터 추가
    for path in false_image_paths:
        dataset.append({"image_path": path, "label": 0})
    
    print(f"데이터셋 구성 완료:")
    print(f"- True 라벨: {len(true_image_paths)}개")
    print(f"- False 라벨: {len(false_image_paths)}개")
    print(f"- 총 데이터셋 크기: {len(dataset)}개")
    
    return dataset

@task
def train_lilt(dataset: list):
    """LiLT 경량 모델 학습 및 검증 (메모리 최적화 버전)"""
    import torch
    from torch.utils.data import Dataset, DataLoader, random_split
    from transformers import AutoProcessor, AutoModelForSequenceClassification
    from torch.optim import AdamW
    from PIL import Image
    import pytesseract
    from torchvision.transforms import Resize
    import os

    os.makedirs(OUTPUT_MODEL_DIR, exist_ok=True)
    if not dataset:
        print("데이터셋이 비어있습니다.")
        return

    device = torch.device("cpu")  # CPU로 고정
    torch.set_default_dtype(torch.float32)

    # LiLT 모델 및 프로세서 로드
    processor = AutoProcessor.from_pretrained(
        "SCUT-DLVCLab/lilt-roberta-en-base",
        use_fast=True
    )
    model = AutoModelForSequenceClassification.from_pretrained(
        "SCUT-DLVCLab/lilt-roberta-en-base",
        num_labels=2,
        torch_dtype=torch.float32
    ).to(device)

    class DocDataset(Dataset):
        def __init__(self, data, processor):
            self.data = data
            self.processor = processor
            self.resize = Resize((224, 224))

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            item = self.data[idx]
            image = Image.open(item["image_path"]).convert("RGB")
            image = self.resize(image)
            try:
                text = pytesseract.image_to_string(image, lang='kor+eng', config='--psm 4 --oem 3')
            except:
                text = ""
            words = text.split()[:50] or ["[UNK]"]
            boxes = [[0,0,100,100]] * len(words)
            encoding = self.processor(
                text=words,
                boxes=boxes,
                return_tensors="pt",
                truncation=True,
                padding="max_length",
                max_length=128
            )
            return {
                **{k: v.squeeze(0) for k, v in encoding.items()},
                "labels": torch.tensor(item["label"], dtype=torch.long)
            }

    # 데이터셋 분할 (훈련 80%, 검증 20%)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    # 데이터로더 생성
    train_loader = DataLoader(
        DocDataset(train_dataset, processor),
        batch_size=2,
        shuffle=True
    )
    val_loader = DataLoader(
        DocDataset(val_dataset, processor),
        batch_size=2,
        shuffle=False
    )

    optimizer = AdamW(model.parameters(), lr=2e-5)

    for epoch in range(3):  # 3 에폭
        print(f"Epoch {epoch+1}")
        # 1. 학습
        model.train()
        train_loss = 0
        for batch in train_loader:
            try:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs, labels=batch["labels"].to(device))
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                train_loss += loss.item()
            except Exception as e:
                print(f"Batch 실패: {str(e)}")
                continue
        print(f"Epoch {epoch+1} 훈련 손실: {train_loss/len(train_loader):.4f}")

        # 2. 검증
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs = {k: v.to(device) for k, v in batch.items() if k != "labels"}
                outputs = model(**inputs, labels=batch["labels"].to(device))
                val_loss += outputs.loss.item()
                preds = torch.argmax(outputs.logits, dim=1)
                correct += (preds == batch["labels"].to(device)).sum().item()
        print(f"Epoch {epoch+1} 검증 손실: {val_loss/len(val_loader):.4f}, 정확도: {correct/len(val_loader):.2%}")

    model.save_pretrained(OUTPUT_MODEL_DIR)
    print(f"모델 저장 완료: {OUTPUT_MODEL_DIR}")

@task
def image_data_augment():
    from torchvision import transforms
    from PIL import Image

    AUG_THRESHOLD = 200
    def get_augmentation():
        return transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.02, 0.02)),
            transforms.ToTensor()
        ])
    def get_image_paths(directory):
        return [os.path.join(directory, fname) for fname in os.listdir(directory) if fname.lower().endswith(('png', 'jpg', 'jpeg'))]

    image_paths = get_image_paths(ORIGIN_TRUE_IMAGE_DIR)
    if len(image_paths) < AUG_THRESHOLD:
        augmentation = get_augmentation() # 증강 함수
        num_aug = 3  # 원본 1장당 증강 이미지 개수
    else:
        augmentation = transforms.ToTensor() # 증강없이 진행
        num_aug = 1 # 원본만
    os.makedirs(READY_TRUE_IMAGE_DIR, exist_ok=True)

    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        base_name = os.path.splitext(os.path.basename(img_path))[0]
        for i in range(num_aug):
            aug_img = augmentation(img)
            aug_img_pil = transforms.ToPILImage()(aug_img)
            save_path = os.path.join(READY_TRUE_IMAGE_DIR, f"{base_name}_aug{i}.png")
            aug_img_pil.save(save_path)


with DAG(
    dag_id='lilt_document_classifier_balanced_V0.1',
    start_date=datetime(2024, 1, 1),
    schedule=None,  # None으로 설정하면 수동 트리거만 가능
    catchup=False,
    tags=['document', 'classification', 'balanced']
) as dag:
    class_create_init_task = init_task()
    # 증강 처리
    image_data_augment_task = image_data_augment()
    # true 파일에 따라 false 파일 채우기
    balance_result_task = balance_false_images(READY_IMAGE_DIR)
    
    #스텝정보 가져오기
    a_class_classify_preprocess_info = file_util.get_step_info_list("a_class","classify","img_preprocess")

    true_file_info_list_task = get_file_info_list_task(READY_TRUE_IMAGE_DIR)
    true_file_preprocess_partial_task = img_preprocess_task.partial(step_info=a_class_classify_preprocess_info,result_key="true")
    true_file_list_task = true_file_preprocess_partial_task.expand(file_info=true_file_info_list_task)
    true_copy_results_task = copy_results_folder_task(true_file_list_task,dest_folder=PREPRC_TRUE_IMAGE_DIR,result_key="true")
    
    false_file_info_list_task = get_file_info_list_task(READY_FALSE_IMAGE_DIR)
    false_file_preprocess_partial_task = img_preprocess_task.partial(step_info=a_class_classify_preprocess_info,result_key="false")
    false_file_list_task = false_file_preprocess_partial_task.expand(file_info=false_file_info_list_task)
    false_copy_results_task = copy_results_folder_task(false_file_list_task,dest_folder=PREPRC_FALSE_IMAGE_DIR,result_key="false")

    # 3. 균형이 맞춰진 데이터셋 구성
    dataset = build_balanced_dataset(READY_IMAGE_DIR)

    # 4. 모델 학습
    train = train_lilt(dataset)
    
    class_create_init_task >> image_data_augment_task
    image_data_augment_task >> balance_result_task >> [true_file_info_list_task, false_file_info_list_task]
    true_file_info_list_task >> true_file_list_task >> true_copy_results_task
    false_file_info_list_task >> false_file_list_task >> false_copy_results_task
    [true_copy_results_task,false_copy_results_task] >> dataset >> train
    # image_data_augment_task >> balance_result_task >> dataset >> train
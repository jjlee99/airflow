from airflow.decorators import task
from collections import Counter
from pathlib import Path
from PIL import Image
import numpy as np
import cv2, os
from typing import Any,List
import uuid
from utils import file_util
from utils import type_convert_util
from airflow.models import Variable,XCom
import pytesseract
from scipy.ndimage import interpolation as inter

RESULT_FOLDER = Variable.get("RESULT_FOLDER", default_var="/opt/airflow/data/result")
TEMP_FOLDER = Variable.get("TEMP_FOLDER", default_var="/opt/airflow/data/temp")
work_in_progress_map = {}  # 작업 중 캐시 정보 관리
result_map = {}           # 최종 결과 파일 경로 관리
@task
def img_preprocess_task(step_info:dict,file_info:dict,result_key:str="result")->dict:
    process_id = str(uuid.uuid4())
    result_map["process_id"] = f"_{process_id}_pre"
    result_map["folder_path"] = str(Path(file_info["file_id"]) / process_id)
    result_map["step_list"] = step_info["step_list"]
    result_map["result_file_map"] = {}
    function_map = {
        #common
        "cache": {"function": cache, "input_type": "file_path", "output_type": "file_path","param":"key"},
        "load": {"function": load, "input_type": "any", "output_type": "file_path","param":"key"},
        "save": {"function": save, "input_type": "file_path", "output_type": "file_path","param":"key"},
        #set
        "calc_angle_set1": {"function": calc_angle_set1, "input_type": "np_bgr", "output_type": "np_bgr","param":"key,iterations,iter_save"},
        "calc_angle_set2": {"function": calc_angle_set2, "input_type": "np_bgr", "output_type": "np_bgr","param":"key,iterations,iter_save"},
        "calc_angle_set3": {"function": calc_angle_set3, "input_type": "np_bgr", "output_type": "np_bgr","param":"key,iterations,iter_save"},
        "calc_angle_set4": {"function": calc_angle_set4, "input_type": "np_bgr", "output_type": "np_bgr","param":"key,iterations,iter_save"},
        "text_orientation_set": {"function": text_orientation_set, "input_type": "np_bgr", "output_type": "np_bgr","param":"key,iterations,iter_save"},
        #preprocess
        "scale1": {"function": scale1, "input_type": "np_bgr", "output_type": "np_bgr"},
        "gray": {"function": gray, "input_type": "np_bgr", "output_type": "np_gray"},
        "denoising1": {"function": denoising1, "input_type": "np_bgr", "output_type": "np_bgr"},
        "denoising2": {"function": denoising2, "input_type": "np_bgr", "output_type": "np_bgr"},
        "threshold": {"function": threshold, "input_type": "np_gray", "output_type": "np_gray"},
        "morphology1": {"function": morphology1, "input_type": "np_bgr", "output_type": "np_bgr"},
        "canny": {"function": canny, "input_type": "np_bgr", "output_type": "np_bgr"},
        "thinner": {"function": thinner, "input_type": "np_bgr", "output_type": "np_bgr"},
        "before_angle1": {"function": before_angle1, "input_type": "np_bgr", "output_type": "np_gray","param":"key"},
        "calc_angle1": {"function": calc_angle1, "input_type": "np_gray", "output_type": "np_gray","param":"key"},
        "before_angle2": {"function": before_orientation, "input_type": "np_bgr", "output_type": "np_gray","param":"key"},
        "calc_angle2": {"function": calc_orientation, "input_type": "any", "output_type": "np_bgr","param":"key"},
        "rotate": {"function": rotate, "input_type": "np_bgr", "output_type": "np_bgr","param":"key"},
        
        "line_tracking": {"function": line_tracking, "input_type": "np_gray", "output_type": "np_gray","param":"iter_save"},
        
    }
    print("empty map check",work_in_progress_map,result_map)
    file_path = file_info["file_path"]
    output = file_path
    before_output_type = "file_path"
    for stepinfo in step_info["step_list"]:
        print("step :",stepinfo["name"])
        function_info = function_map[stepinfo["name"]]
        convert_param = stepinfo.get("convert_param", {})
        input = type_convert_util.convert_type(output,before_output_type,function_info["input_type"],params=convert_param)
        output = function_info["function"](input,**stepinfo["param"])
        before_output_type = function_info["output_type"]
    file = type_convert_util.convert_type(output,"np_bgr","file_path")
    save(file,f"_result")
    file_info[result_key]=result_map
    return file_info

def cache(file_path:str,key:str)->str:
    work_in_progress_map[f"filepath_{key}"] = file_path
    return file_path

def load(_,key:str)->str:
    return work_in_progress_map[f"filepath_{key}"]

def save(file_path:str,key:str)->str:
    save_path = Path(TEMP_FOLDER) / result_map["folder_path"] / f"{key}.png"
    save_path = file_util.file_copy(file_path,save_path)
    result_map["result_file_map"][key]=save_path
    return file_path

def scale1(img_np_bgr:np.ndarray,width:int,height:int) -> np.ndarray:
    """
    이미지를 지정한 크기(width, height)에 맞게 비율을 유지하며 리사이즈하고,
    남는 공간은 흰색(255)으로 채웁니다.

    :param img_np_bgr: BGR 채널을 가진 numpy 배열(OpenCV 이미지)
    :param width: 결과 이미지의 가로 크기
    :param height: 결과 이미지의 세로 크기
    :return: 크기 조정 및 중앙 정렬된 BGR numpy 배열
    """
    original_height, original_width = img_np_bgr.shape[:2]

    # 비율을 유지하면서 목표 크기를 넘지 않는 최대 크기 계산
    scale_ratio = min(width/original_width, height/original_height)

    # 새로운 크기 계산
    new_width = int(original_width * scale_ratio)
    new_height = int(original_height * scale_ratio)
    
    if scale_ratio < 1.0:
        resized_image = cv2.resize(img_np_bgr, (new_width, new_height), interpolation=cv2.INTER_AREA)
    else:
        resized_image = cv2.resize(img_np_bgr, (new_width, new_height), interpolation=cv2.INTER_CUBIC)
    background = np.full((height, width, 3), 255, dtype=np.uint8)

    # 리사이징된 이미지를 하얀색 배경 중앙에 붙여넣기
    paste_x = (width - new_width) // 2
    paste_y = (height - new_height) // 2
    background[paste_y:paste_y+new_height, paste_x:paste_x+new_width] = resized_image

    return background

def gray(img_np_bgr: np.ndarray) -> np.ndarray:
    """
    이미지를 그레이스케일로 변환합니다.

    :param img_np_bgr: BGR 채널을 가진 numpy 배열(OpenCV 이미지)
    :return: 그레이스케일로 변환된 numpy 배열
    """
    return cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY)

def denoising1(img_np_bgr: np.ndarray) -> np.ndarray:
    """
    컬러 이미지에 대해 Non-Local Means 알고리즘으로 노이즈를 제거합니다.

    :param img_np_bgr: BGR 채널을 가진 numpy 배열(OpenCV 이미지)
    :return: 노이즈가 제거된 BGR numpy 배열
    """
    denoised_np_bgr = cv2.fastNlMeansDenoisingColored(
        src=img_np_bgr,
        h=3,                  # 밝기 성분 강도
        hColor=3,             # 색상 성분 강도
        templateWindowSize=7, # 검사 패치 크기
        searchWindowSize=21   # 검색 윈도우 크기
    )
    return denoised_np_bgr

def denoising2(img_np_bgr: np.ndarray) -> np.ndarray:
    """
    컬러 이미지에 대해 미디언 블러(median blur)로 노이즈를 제거합니다.

    :param img_np_bgr: BGR 채널을 가진 numpy 배열(OpenCV 이미지)
    :return: 노이즈가 제거된 BGR numpy 배열
    """
    denoised_np_bgr = cv2.medianBlur(img_np_bgr, 3)
    return denoised_np_bgr

def threshold(img_np_gray:np.ndarray,thresh:int=127,type:int=cv2.THRESH_BINARY) -> np.ndarray:
    """
    이미지를 임계값 기준으로 이진화합니다.
    thresh : 임계값
    type : 다음 코드값에 따라 지정된 작업을 실행합니다.
      cv2.THRESH_BINARY : 임계값보다 크면 255(흰색) 작으면 0(검정)
      cv2.THRESH_BINARY_INV : 임계값보다 크면 0(검정) 작으면 255(흰색)
      cv2.THRESH_TOZERO : 임계값 이하만 0(검정) 그 외 현상유지
      cv2.THRESH_TOZERO_INV : 임계값 이상만 0(검정) 그 외 현상유지
      cv2.THRESH_OTSU : 이 옵션을 추가하면 임계값을 자동 지정함
    """
    ret, binary_np_gray = cv2.threshold(img_np_gray, thresh=thresh, maxval=255, type=type)
    return binary_np_gray

def adaptive_threshold(img_np_gray:np.ndarray,type:int=cv2.THRESH_BINARY,block:int=11) -> np.ndarray:
    """
    국소적 자동 임계값을 기준으로 이진화합니다.
    type : 다음 코드값에 따라 지정된 작업을 실행합니다.
      cv2.THRESH_BINARY : 임계값보다 크면 255(흰색) 작으면 0(검정)
      cv2.THRESH_BINARY_INV : 임계값보다 크면 0(검정) 작으면 255(흰색)
      cv2.THRESH_TOZERO : 임계값 이하만 0(검정) 그 외 현상유지
      cv2.THRESH_TOZERO_INV : 임계값 이상만 0(검정) 그 외 현상유지
    block : 작을수록 잡음 제거력이 떨어지며, 클수록 이미지 뭉개짐이 높아집니다
    """
    binary_np_gray = cv2.adaptiveThreshold(
        img_np_gray, maxValue=255,
        adaptiveMethod=cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        thresholdType=type,
        blockSize=block, C=2
    )
    return binary_np_gray

def morphology1(img_np_bgr: np.ndarray) -> np.ndarray:
    """
    형태학적 연산(열기 → 닫기)을 적용해 노이즈를 제거하고 객체 경계를 보정합니다.

    :param img_np_bgr: 이진화된 numpy 배열(OpenCV 이미지)
    :return: 형태학적 연산이 적용된 numpy 배열
    """
    kernel = np.ones((3,3), np.uint8) #커널은 중앙 홀수로 작업
    open_img_np_bgr = cv2.morphologyEx(img_np_bgr, cv2.MORPH_OPEN, kernel)
    morphology_img_bin = cv2.morphologyEx(open_img_np_bgr, cv2.MORPH_CLOSE, kernel)
    return morphology_img_bin

def canny(img_np_bgr: np.ndarray) -> np.ndarray:
    # 엣지 검출
    edges = cv2.Canny(img_np_bgr, 30, 100, apertureSize=3)
    return edges

def thinner(img_np_bgr: np.ndarray) -> np.ndarray:
    # 전처리 예시: 선 굵기 줄이기
    kernel = np.ones((3,3), np.uint8)
    edges = cv2.erode(img_np_bgr, kernel, iterations=1)
    return edges

def calc_angle_set1(img_np_bgr: np.ndarray,key:str,iterations:int=3,iter_save:bool=False) -> np.ndarray:
    """
    다각형 근사화를 활용한 표 인식 및 미세회전
    문서 방향 조정을 위해 text_orientation_set과 함께 사용 추천
    """
    target_img=img_np_bgr
    total_angle=0
    idx=1
    while idx<=iterations:
        # 1. 회전을 위한 전처리
        # 1-1. 그레이스케일
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        # 1-2. 이진화 (표 경계를 명확하게)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # # 임시 저장
        # if iter_save:
        #     file = type_convert_util.convert_type(thresh,"np_bgr","file_path")
        #     save(file,f"rotate1_{idx}")
        
        # 2. 보정각도 추출
        # 2-1. 윤곽선 검출
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        # 2-2. 가장 큰 윤곽선 선택
        largest_contour = max(contours, key=cv2.contourArea)
        # 2-3. 다각형 근사화 (4꼭지점 추출)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if not len(approx) == 4:
            break
        else:
            # 2-4. 꼭지점 기준 각도 계산(상단만 체크)
            corners = approx.reshape(4, 2)
            # 꼭지점 정렬 (x+y 기준으로 좌상단, 우상단, 좌하단, 우하단)
            sorted_corners = sorted(corners, key=lambda pt: (pt[0] + pt[1]))
            top_left, top_right, bottom_left, bottom_right = sorted_corners
            # 상단 벡터 (우상단 - 좌상단)
            dx_top = top_right[0] - top_left[0]
            dy_top = top_right[1] - top_left[1]
            angle_top = np.degrees(np.arctan2(dy_top, dx_top))
            # 하단 벡터 (우상단 - 좌상단)
            dx_top = bottom_right[0] - bottom_left[0]
            dy_top = bottom_right[1] - bottom_left[1]
            angle_bottom = np.degrees(np.arctan2(dy_top, dx_top))
            # 좌측 벡터 (좌하단 - 좌상단)
            dx_left = bottom_left[0] - top_left[0]
            dy_left = bottom_left[1] - top_left[1]
            angle_left = np.degrees(np.arctan2(dy_left, dx_left)) + 90
            # 우측 벡터 (좌하단 - 좌상단)
            dx_left = bottom_right[0] - top_right[0]
            dy_left = bottom_right[1] - top_right[1]
            angle_right = np.degrees(np.arctan2(dy_left, dx_left)) + 90
            # 평균
            avg_angle = (angle_top+angle_bottom+angle_left+angle_right)/4
            print(f"angle{idx} : ",total_angle,avg_angle, angle_left, angle_right, angle_top, angle_bottom)
            if avg_angle < 0.1:
                break
            
            # 3. 타겟이미지를 보정 각도만큼 회전
            rotated = _rotate(target_img,avg_angle)
            # 4. 반복 처리를 위한 작업
            total_angle+=avg_angle
            target_img = rotated
            idx+=1
    
    work_in_progress_map[f"angle_{key}"] = total_angle
    return target_img


def before_angle1(img_np_bgr: np.ndarray) -> np.ndarray:
    # 1. 그레이스케일
    gray = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY)
    # 2. 이진화 (표 경계를 명확하게)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def calc_angle1(img_np_gray: np.ndarray, key: str) -> float:
    # 윤곽선 검출
    contours, _ = cv2.findContours(img_np_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 가장 큰 윤곽선 선택
    largest_contour = max(contours, key=cv2.contourArea)
    # 다각형 근사화 (4꼭지점 추출)
    epsilon = 0.02 * cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, epsilon, True)
    
    if not len(approx) == 4:
        print("표의 4꼭지점을 찾을 수 없습니다.")
        work_in_progress_map[f"angle_{key}"] = 0
        return img_np_gray
    else:
        corners = approx.reshape(4, 2)
        # 꼭지점 정렬 (x+y 기준으로 좌상단, 우상단, 좌하단, 우하단)
        sorted_corners = sorted(corners, key=lambda pt: (pt[0] + pt[1]))
        top_left, top_right, bottom_left, bottom_right = sorted_corners

        # 상단 벡터 (우상단 - 좌상단)
        dx_top = top_right[0] - top_left[0]
        dy_top = top_right[1] - top_left[1]
        angle_top = np.degrees(np.arctan2(dy_top, dx_top))
        # 가로선을 0도로 맞추는 보정 각도
        correction_angle_top = -angle_top

        # 좌측 벡터 (좌하단 - 좌상단)
        dx_left = bottom_left[0] - top_left[0]
        dy_left = bottom_left[1] - top_left[1]
        angle_left = np.degrees(np.arctan2(dy_left, dx_left))
        # 세로선을 90도로 맞추는 보정 각도
        correction_angle_left = 90 - angle_left

        # 보정 각도 출력 및 반환 (가로선 기준 또는 평균, 필요에 따라 선택)
        print(f"가로선 보정 각도: {correction_angle_top:.2f}도")
        print(f"세로선 보정 각도: {correction_angle_left:.2f}도")
        work_in_progress_map[f"angle_{key}"] = (correction_angle_top*correction_angle_left)/2 * -1
        return img_np_gray

def calc_angle_set2(img_np_bgr:np.ndarray,key:str,iterations:int=3,iter_save:bool=False) -> np.ndarray:
    """
    각도별 커널 탐색을 활용한 수평선/수직선 인식 및 미세회전
    문서 방향 조정을 위해 text_orientation_set과 함께 사용 추천
    """
    target_img=img_np_bgr
    total_angle=0
    idx=1
    
    while idx<=iterations:
        delta = 0.25
        limit = 5
        angles = np.arange(-limit, limit + delta, delta)
        scores = []
        
        (h, w) = target_img.shape[:2]
        min_length = int(min(w,h) * 0.1)
        min_length2 = int(min(w,h) * 0.4)

        def long_kernal_score(arr, angle):
            i=0
            horizon_kernel = np.ones((min_length, 1), np.uint8)
            vertical_kernel = np.ones((1, min_length), np.uint8)
            horizon_kernel2 = np.ones((min_length2, 1), np.uint8)
            vertical_kernel2 = np.ones((1, min_length2), np.uint8)
            
            # # 1-3. 작은 객체 제거
            # kernel_horiz = np.ones((1, line_kernel), np.uint8)  # 가로 방향 커널 (길이 조정 필요)
            # horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horiz)

            # # 수직선 강조 (글자 제거, 수직선만 남기기)
            # kernel_vert = np.ones((line_kernel, 1), np.uint8)   # 세로 방향 커널 (길이 조정 필요)
            # vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vert)
            
            # # 수평+수직 합치기
            # line_filtered = cv2.add(horizontal, vertical)
            
            # # 1.3. 반전
            # lines_inv = cv2.bitwise_not(line_filtered)
            
            # # 1-4. 에지 검출 (Canny)
            # edges = cv2.Canny(lines_inv, 50, 150, apertureSize=3)
        
            #3,3으로 팽창
            dilated = cv2.dilate(arr, np.ones((3, 3), np.uint8), iterations=1)
            
            # data = inter.rotate(arr, angle, reshape=False, order=0)
            data = _rotate(dilated,angle)
            
            # 1-1. 그레이스케일
            gray = cv2.cvtColor(data, cv2.COLOR_BGR2GRAY)
            # 1-2. 이진화 (표 경계를 명확하게)
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
                        
            horizon_eroded = cv2.erode(thresh, horizon_kernel)
            vertical_eroded = cv2.erode(thresh, vertical_kernel)
            horizon_eroded2 = cv2.erode(thresh, horizon_kernel2)
            vertical_eroded2 = cv2.erode(thresh, vertical_kernel2)
            
            score = cv2.countNonZero(horizon_eroded) + cv2.countNonZero(vertical_eroded) + cv2.countNonZero(horizon_eroded2) + cv2.countNonZero(vertical_eroded2)
            # # 임시 저장 
            # if iter_save :
            #     tmp=cv2.add(horizon_eroded,vertical_eroded)
            #     tmp2=cv2.add(horizon_eroded2,vertical_eroded2)
            #     tmp3=cv2.add(tmp,tmp2)
            #     i+=1
            #     file = type_convert_util.convert_type(tmp3,"np_bgr","file_path")
            #     save(file,f"rotate11_{idx}_{angle}_{score}")
            
            return score
        for angle in angles:
            score = long_kernal_score(target_img, angle)
            scores.append(score)
        
        threshold_val = 50
        best_angle = angles[scores.index(max(scores))]
        if max(scores) <= threshold_val:
            best_angle = 0
        total_angle+=best_angle
        print(f"osd {idx}",total_angle,best_angle,scores.index(max(scores)), max(scores), scores)
        
        # 3. 타겟이미지를 보정 각도만큼 회전
        rotated = _rotate(target_img,best_angle)
    
        # 4. 반복 처리를 위한 작업
        target_img = rotated
        idx+=1
    work_in_progress_map[f"angle_{key}"] = total_angle
    return target_img 

def calc_angle_set3(img_np_bgr:np.ndarray,key:str,iterations:int=3,iter_save:bool=False) -> np.ndarray:
    """
    허프변환을 활용한 직선 인식 및 미세회전
    문서에 따른 수치 조정이 많이 필요함(미완)
    문서 방향 조정을 위해 text_orientation_set과 함께 사용 추천
    """
    target_img=img_np_bgr
    total_angle=0
    idx=1
    tolerance = 2
    
    height, width = img_np_bgr.shape[:2]
    line_kernel = int(min(width,height) * 0.2)  # 전체 크기 10% 이상 길이의 선을 기준으로 문자 제거
    while idx<=iterations:
        # 1-1. 그레이스케일
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        # 1-2. 이진화 (표 경계를 명확하게)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # 1-3. 작은 객체 제거
        kernel_horiz = np.ones((1, line_kernel), np.uint8)  # 가로 방향 커널 (길이 조정 필요)
        horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_horiz)

        # 수직선 강조 (글자 제거, 수직선만 남기기)
        kernel_vert = np.ones((line_kernel, 1), np.uint8)   # 세로 방향 커널 (길이 조정 필요)
        vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_vert)
        
        # 수평+수직 합치기
        line_filtered = cv2.add(horizontal, vertical)
        
        # 1.3. 반전
        lines_inv = cv2.bitwise_not(line_filtered)
        
        # 1-4. 에지 검출 (Canny)
        edges = cv2.Canny(lines_inv, 50, 150, apertureSize=3)
        
        # 임시 저장
        if iter_save:
            file = type_convert_util.convert_type(edges,"np_bgr","file_path")
            save(file,f"rotate3_{idx}")
        
        # 2. Hough Line Transform으로 선 검출
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=150)
        angles = []
        if lines is not None:
            for line in lines:
                rho, theta = line[0]
                # 각도를 도 단위로 변환
                angle = np.degrees(theta)
                # 0~179도 범위로 정규화
                angle = angle % 180
                if angle < 0:
                    angle += 180
                angles.append(angle)
        
        if not angles:
            print("선을 찾을 수 없습니다.")
            break
        grouped_angles = []
        for angle in angles:
            grouped_angle = round(angle / tolerance) * tolerance
            grouped_angles.append(grouped_angle)
        tolerance = tolerance/2
        # 가장 많이 나타나는 각도 찾기
        angle_counter = Counter(grouped_angles)
        dominant_angle = angle_counter.most_common(1)[0][0]
        
        # 후보군 중 가장 적게 회전하는 각도 탐색
        candidates = [0, 90, -90]
        differences = [abs(dominant_angle - candidate) for candidate in candidates]
        target_angle = candidates[differences.index(min(differences))]
        rotation_angle = target_angle - dominant_angle

        total_angle+=rotation_angle
        print(f"osd {idx}",total_angle,rotation_angle, angles)
            
        # 3. 타겟이미지를 보정 각도만큼 회전
        rotated = _rotate(target_img,rotation_angle)
        
        # 4. 반복 처리를 위한 작업
        target_img = rotated
        idx+=1
    work_in_progress_map[f"angle_{key}"] = total_angle
    return target_img

def calc_angle_set4(img_np_bgr:np.ndarray,key:str,iterations:int=3,iter_save:bool=False) -> np.ndarray:
    """
    각도별 수평/수직 픽셀들의 변화 활용해 미세회전
    문서 방향 조정을 위해 text_orientation_set과 함께 사용 추천
    """
    target_img=img_np_bgr
    total_angle=0
    idx=1
    
    while idx<=iterations:
        # 1-1. 그레이스케일
        gray = cv2.cvtColor(target_img, cv2.COLOR_BGR2GRAY)
        # 1-2. 이진화 (표 경계를 명확하게)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

        delta = 0.25
        limit = 5
        angles = np.arange(-limit, limit + delta, delta)
        scores = []

        def histogram_score(arr, angle):
            data = inter.rotate(arr, angle, reshape=False, order=0)
            histogram = np.sum(data, axis=1, dtype=float)
            score = np.sum((histogram[1:] - histogram[:-1]) ** 2, dtype=float)
            return score
        
        for angle in angles:
            score = histogram_score(thresh, angle)
            scores.append(score)

        best_angle = angles[scores.index(max(scores))]
        total_angle+=best_angle
        print(f"osd {idx}",total_angle,best_angle,scores)
        
        # 3. 타겟이미지를 보정 각도만큼 회전
        rotated = _rotate(target_img,best_angle)
        
        # 4. 반복 처리를 위한 작업
        target_img = rotated
        idx+=1
    work_in_progress_map[f"angle_{key}"] = total_angle
    return target_img

def text_orientation_set(img_np_bgr:np.ndarray,key:str,iterations:int=2,iter_save:bool=False) -> np.ndarray:
    """
    테서랙트의 텍스트 방향과 문자 종류 감지를 활용한 90도 단위 회전
    미세조정을 위해 calc_angle_set1,3,5 등과 함께 사용 추천
    """
    target_img=img_np_bgr
    total_angle=0
    idx=1
    while idx<=iterations:
        # 1. 회전을 위한 전처리
        # 1-1. 노이즈 제거
        denoised = cv2.fastNlMeansDenoisingColored(
            src=target_img,
            h=3,                  # 밝기 성분 강도
            hColor=3,             # 색상 성분 강도
            templateWindowSize=7, # 검사 패치 크기
            searchWindowSize=21   # 검색 윈도우 크기
        )
        # 1-2. 그레이스케일
        gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
        # 1-3. 이진화 (표 경계를 명확하게)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 1-4. 태서렉트 입력을 위해 rgb로 변경
        rgb = cv2.cvtColor(binary, cv2.COLOR_GRAY2RGB)
        # 임시 저장
        if iter_save:
            file = type_convert_util.convert_type(rgb,"np_bgr","file_path")
            save(file,f"rotate2_{idx}")
        
        # 2. ocr을 이용한 보정 각도 추출
        try:
            osd = pytesseract.image_to_osd(rgb)
        except pytesseract.TesseractError as e:
            print(f"Tesseract OSD Error: {e}")
            break
    
        rotation = 0
        for info in osd.split('\n'):
            if 'Rotate: ' in info:
                rotation = int(info.split(': ')[1])
            if 'Orientation confidence:' in info:
                orientation_confidence = float(info.split(': ')[1])
            if 'Script: ' in info:
                script_name = info.split(': ')[1]
            if 'Script confidence:' in info:
                script_confidence = float(info.split(': ')[1])
        if rotation == 0:
            print(f"osd {idx} break ",total_angle,rotation, orientation_confidence, script_name, script_confidence)
            break
        total_angle+=rotation
        print(f"osd {idx}",total_angle,rotation, orientation_confidence, script_name, script_confidence)
        
        # 3. 타겟이미지를 보정 각도만큼 회전
        rotated = _rotate(target_img,rotation)
        
        # 4. 반복 처리를 위한 작업
        target_img = rotated
        idx+=1
    work_in_progress_map[f"angle_{key}"] = total_angle
    return target_img

def before_orientation(img_np_bgr: np.ndarray) -> np.ndarray:
    # 1. 그레이스케일
    gray = cv2.cvtColor(img_np_bgr, cv2.COLOR_BGR2GRAY)
    # 2. 이진화 (표 경계를 명확하게)
    _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    return thresh

def calc_orientation(img_np_bgr: np.ndarray,key:str) -> np.ndarray:
    """테서랙트 OSD를 이용한 방향 보정"""
    try:
        osd = pytesseract.image_to_osd(img_np_bgr)
        rotation = 0
        for info in osd.split('\n'):
            if 'Rotate: ' in info:
                rotation = int(info.split(': ')[1])
    except pytesseract.TesseractError as e:
        print(f"Tesseract OSD Error: {e}")
        rotation = 0            
    print(f"가로선 보정 각도: {rotation:.2f}도")
    work_in_progress_map[f"angle_{key}"] = rotation
    return img_np_bgr


def rotate(img_np_bgr: np.ndarray,key:str) -> np.ndarray:
    """이미지 회전 함수"""
    angle = work_in_progress_map[f"angle_{key}"]
    if angle == 0:
        return img_np_bgr
    rotated = _rotate(img_np_bgr,angle)
    return rotated

def line_tracking(img_np_gray:np.ndarray, iter_save:bool=False) -> np.ndarray:
    _, binary = cv2.threshold(img_np_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # 임시 저장
    if iter_save:
        print("binary:", binary.shape)
        file = type_convert_util.convert_type(binary,"np_gray","file_path")
        save(file,f"line_binary")
    
    # 2. 수평/수직 라인 강조 (모폴로지 연산)
    horizontal = binary.copy()
    vertical = binary.copy()

    # 수평 라인 검출
    cols = horizontal.shape[1]  #가로픽셀수
    horizontal_size = cols // 30  # 표 구조에 따라 조정
    horizontal_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (horizontal_size, 1))
    horizontal = cv2.erode(horizontal, horizontal_structure)
    horizontal = cv2.dilate(horizontal, horizontal_structure)

    # 수직 라인 검출
    rows = vertical.shape[0]  #세로픽셀수
    vertical_size = rows // 30  # 표 구조에 따라 조정
    vertical_structure = cv2.getStructuringElement(cv2.MORPH_RECT, (1, vertical_size))
    vertical = cv2.erode(vertical, vertical_structure)
    vertical = cv2.dilate(vertical, vertical_structure)

    # 3. 라인 합치기 (표 전체 구조)
    table_mask = cv2.add(horizontal, vertical)

    # 4. 라인 추적 (연결성 보완)
    # 라벨링으로 연결된 라인 추출
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(table_mask, connectivity=8)

    # 결과 시각화
    output = cv2.cvtColor(img_np_gray, cv2.COLOR_GRAY2BGR)
    for i in range(1, num_labels):
        x, y, w, h, area = stats[i]
        if area > 50:  # 작은 잡음 제거
            cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # 임시 저장
    if iter_save:
        print("output:", output.shape)
        file = type_convert_util.convert_type(output,"np_bgr","file_path")
        save(file,f"detected_lines")
    return output
    
    

#내부 함수
def _rotate(img_np_bgr: np.ndarray,angle:float) -> np.ndarray:
    """이미지 회전 내부 함수"""
    h, w = img_np_bgr.shape[:2]
    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img_np_bgr, M, (w, h), flags=cv2.INTER_CUBIC, 
                        borderMode=cv2.BORDER_REPLICATE)
    
    # 회전 후 이미지 크기 계산
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 회전 중심 조정 (중심 이동)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    rotated = cv2.warpAffine(
        img_np_bgr, M, (new_w, new_h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated


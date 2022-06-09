import numpy as np
import pandas as pd
import os
import random
#logging 도구 import 하기
import logging

def zero_min_row(zero_matrix, mark_zero):
    min_row = [99999, -1]

    #0이 가장 적은 행 찾기(True가 가장 적은 행 찾기)
    for i in range(zero_matrix.shape[0]):
        if np.sum(zero_matrix[i] == True) > 0 and min_row[0] > np.sum(zero_matrix[i] == True):
            min_row = [np.sum(zero_matrix[i] == True), i]

    #True가 가장 적은 행(0이 가장 적은 행)의 모든 요소를 False로 처리
    zero_index = np.where(zero_matrix[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1], zero_index))
    zero_matrix[min_row[1], :] = False
    zero_matrix[:, zero_index] = False


def mark_matrix(matrix):
    #0이 가장 적은 행을 찾기 위해 matrix를 boolean matrix로 변환: 0만 True, 나머지 숫자는 False
    cur_matrix = matrix
    zero_bool_matrix = (cur_matrix == 0)
    zero_bool_matrix_copy = zero_bool_matrix.copy()

    marked_zero = []
    while (True in zero_bool_matrix_copy):
        zero_min_row(zero_bool_matrix_copy, marked_zero)

    # 행과 열 분리하여 기록
    marked_zero_row = []
    marked_zero_col = []
    for i in range(len(marked_zero)):
        marked_zero_row.append(marked_zero[i][0])
        marked_zero_col.append(marked_zero[i][1])

    #0을 포함하지 않는 행(행의 요소가 모두 False인 행) 표시, non_marked_row에 행 인덱스 저장
    non_marked_row = list(set(range(cur_matrix.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_matrix[non_marked_row[i], :]
            for j in range(row_array.shape[0]):
                #non_markes_row의 요소 중에 표시되지 않은 0(표시되지 않은 True)이 있는지 확인
                if row_array[j] == True and j not in marked_cols:
                    #marked_cols에 열 인덱스 저장
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            #marked_zero와 marked_cols에 저장된 열 인덱스 비교
            if row_num not in non_marked_row and col_num in marked_cols:
                #일치하는 열 인덱스가 존재한다면 해당 row_index는 non_marked_rows에 저장
                non_marked_row.append(row_num)
                check_switch = True

    #non_marked_row에 없는 행 인덱스는 marker_rows에 저장
    marked_rows = list(set(range(matrix.shape[0])) - set(non_marked_row))

    return (marked_zero, marked_rows, marked_cols)


def adjust_matrix(matrix, cover_rows, cover_cols):
    cur_matrix = matrix
    non_zero_element = []

    #marked_rows에 없고 marked_cols에도 없는 요소 중 최소 비용 찾기
    for row in range(len(cur_matrix)):
        if row not in cover_rows:
            for i in range(len(cur_matrix[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_matrix[row][i])
    min_num = min(non_zero_element)

    #marked_rows와 marked_cols이 아닌 행과 열에서 위에서 찾은 최소 비용 감산하기
    for row in range(len(cur_matrix)):
        if row not in cover_rows:
            for i in range(len(cur_matrix[row])):
                if i not in cover_cols:
                    cur_matrix[row,i] = cur_matrix[row,i] - min_num

    #marked_rows에 있고 marked_cols에도 있는 요소에 위에서 찾은 최소 비용 가산하기
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_matrix[cover_rows[row], cover_cols[col]] = cur_matrix[cover_rows[row], cover_cols[col]] + min_num

    return cur_matrix


def hungarian_algorithm(matrix):
    dim = matrix.shape[0]
    cur_matrix = matrix

    # 각 행, 각 열의 비용에서 최소 비용을 감산하기
    for i in range(dim):
        cur_matrix[i] = cur_matrix[i] - np.min(cur_matrix[i])
    for j in range(dim):
        cur_matrix[:, j] = cur_matrix[:, j] - np.min(cur_matrix[:, j])
    zero_cnt = 0
    while zero_cnt < dim:
        #marked_rows와 marked_cols의 길이 합이 n과 같다면 성공적으로 해를 찾은 것
        #이 때 marked_zero는 솔루션 좌표를 저장
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_matrix)
        zero_cnt = len(marked_rows) + len(marked_cols)
        if zero_cnt < dim:
            cur_matrix = adjust_matrix(cur_matrix, marked_rows, marked_cols)

    return ans_pos


def ans_calculation(matrix, pos):
    total = 0
    ans_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(len(pos)):
        total += matrix[pos[i][0], pos[i][1]]
        ans_matrix[pos[i][0], pos[i][1]] = matrix[pos[i][0], pos[i][1]]
    return total, ans_matrix


def makedirs(path):
    try:
        os.makedirs(path)
    except OSError:
        if not os.path.isdir(path):
            raise


def main():

    # logger 설정하기
    logger = logging.getLogger("main")
    stream_hander = logging.StreamHandler()
    logger.addHandler(stream_hander)
    # logger level 'INFO' 로 설정하기
    logger.setLevel(logging.INFO)
    # logger 출력되는 형식 설정하기
    formatter = logging.Formatter('[%(asctime)s][%(levelname)s|%(filename)s:%(lineno)s] >>> %(message)s')

    N = int(input("Enter an integer n for generating nxn data: "))
    matrix = []

    logging.info('')
    logger.info('making data')

    for i in range(N):
        m = []
        for c in range(N):
            m.append(random.randrange(100))
        matrix.append(m)
    arr = np.array(matrix)
    print(arr)  # 역할1 박채현 데이터 생성

    ans_pos = hungarian_algorithm(arr.copy())
    ans, ans_matrix = ans_calculation(arr, ans_pos)

    logging.info('')
    logger.info('print solution')

    # 솔루션 출력
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_matrix}")
    makedirs('./output')
    df = pd.DataFrame(ans_matrix)
    df.to_csv("./output/result.csv", header=None, index=None)


if __name__ == '__main__':
    main()
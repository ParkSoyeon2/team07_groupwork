# 샘플 Python 스크립트입니다.

# Shift+F10을(를) 눌러 실행하거나 내 코드로 바꿉니다.
# 클래스, 파일, 도구 창, 액션 및 설정을 어디서나 검색하려면 Shift 두 번을(를) 누릅니다.


def print_hi(name):
    # 스크립트를 디버그하려면 하단 코드 줄의 중단점을 사용합니다.
    print(f'Hi, {name}')  # 중단점을 전환하려면 Ctrl+F8을(를) 누릅니다.


# 스크립트를 실행하려면 여백의 녹색 버튼을 누릅니다.
if __name__ == '__main__':
    print_hi('PyCharm')

# https://www.jetbrains.com/help/pycharm/에서 PyCharm 도움말 참조


import numpy as np

#0이 가장 적은 행 찾기, bool matrix로 변환하여 True 찾기
def zero_min_row(zero_matrix, mark_zero):
    min_row = [99999, -1]

    for i in range(zero_matrix.shape[0]):
        if np.sum(zero_matrix[i] == True) > 0 and min_row[0] > np.sum(zero_matrix[i] == True):
            min_row = [np.sum(zero_matrix[i] == True), i]

    zero_index = np.where(zero_matrix[min_row[1]] == True)[0][0]
    mark_zero.append((min_row[1],zero_index))
    zero_matrix[min_row[1],:] = False
    zero_matrix[:,zero_index] = False


def mark_matrix(matrix):
    #boolean matrix로 변환: 0만 True, 나머지 숫자는 False
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


    non_marked_row = list(set(range(cur_matrix.shape[0])) - set(marked_zero_row))

    marked_cols = []
    check_switch = True
    while check_switch:
        check_switch = False
        for i in range(len(non_marked_row)):
            row_array = zero_bool_matrix[non_marked_row[i],:]
            for j in range(row_array.shape[0]):
                if row_array[j] == True and j not in marked_cols:
                    marked_cols.append(j)
                    check_switch = True

        for row_num, col_num in marked_zero:
            if row_num not in non_marked_row and col_num in marked_cols:
                non_marked_row.append(row_num)
                check_switch = True
    marked_rows = list(set(range(matrix.shape[0])) - set(non_marked_row))

    return (marked_zero, marked_rows, marked_cols)


def adjust_matrix(matrix, cover_rows, cover_cols):
    cur_matrix = matrix
    non_zero_element = []

    for row in range(len(cur_matrix)):
        if row not in cover_rows:
            for i in range(len(cur_matrix[row])):
                if i not in cover_cols:
                    non_zero_element.append(cur_matrix[row][i])
    min_num = min(non_zero_element)

    for row in range(len(cur_matrix)):
        if row not in cover_rows:
            for i in range(len(cur_matrix[row])):
                if i not in cover_cols:
                    cur_matrix[row,i] = cur_matrix[row,i] - min_num
    for row in range(len(cover_rows)):
        for col in range(len(cover_cols)):
            cur_matrix[cover_rows[row], cover_cols[col]] = cur_matrix[cover_rows[row],cover_cols[col]] + min_num

    return cur_matrix


def hungarian_algorithm(matrix):
    dim = matrix.shape[0]
    cur_matrix = matrix

    # Step 1: 각 행, 각 열의 비용에서 최소 비용을 감산하기
    for i in range(dim):
        cur_matrix[i] = cur_matrix[i] - np.min(cur_matrix[i])
    for j in range(dim):
        cur_matrix[:,j] = cur_matrix[:,j] - np.min(cur_matrix[:,j])
    zero_cnt = 0
    while zero_cnt < dim:
        ans_pos, marked_rows, marked_cols = mark_matrix(cur_matrix)
        zero_cnt = len(marked_rows) + len(marked_cols)
        if zero_cnt < dim:
            cur_matrix = adjust_matrix(cur_matrix, marked_rows, marked_cols)

    return ans_pos


def ans_calculation(matrix, pos):
    total = 0
    ans_matrix = np.zeros((matrix.shape[0], matrix.shape[1]))
    for i in range(len(pos)):
        total += matrix[pos[i][0],pos[i][1]]
        ans_matrix[pos[i][0], pos[i][1]] = matrix[pos[i][0], pos[i][1]]
    return total, ans_matrix


def main():
    import random

    N = int(input())
    matrix = []
    for i in range(N):
        m = []
        for c in range(N):
            m.append(random.randrange(100))
        matrix.append(m)
    arr = np.array(matrix)
    print(arr)  # 역할1 박채현 데이터 생성

    ans_pos = hungarian_algorithm(arr.copy())
    ans, ans_matrix = ans_calculation(arr, ans_pos)

    # Show the result
    print(f"Linear Assignment problem result: {ans:.0f}\n{ans_matrix}")

if __name__ == '__main__':
    main()
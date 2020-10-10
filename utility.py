import cv2
import numpy as np

def get_rgb_distance(pixel1, pixel2):
    '''
    pixel1, 2 : 3차원 list with rgb value ( 10 진법 ).
    단순한 rgb 값의 euclidean distance를 return 한다.
    '''
    distance = 0
    for i in range(3):
        distance += abs(pixel1[i] - pixel2[i]) ** 2

    return distance ** 1/2

def get_cielab_distance(pixel1, pixel2):
    '''
    pixel1, 2 : 3차원 list with rgb value ( 10 진법 ).
    rgb 값의 red mean color distance를 return 한다.
    '''
    npPixel1 = np.array([[pixel1]], dtype='uint8')
    lab1 = cv2.cvtColor(npPixel1, cv2.COLOR_RGB2Lab).tolist()[0][0]
    npPixel2 = np.array([[pixel2]], dtype='uint8')
    lab2 = cv2.cvtColor(npPixel2, cv2.COLOR_RGB2Lab).tolist()[0][0]

    return get_rgb_distance(lab1, lab2)
    
def get_color_distance_map(class_color, class_length, distance_func=get_cielab_distance):
    '''
    class_color : each class average colors list.
    return : Color distance map for input class_color.
    '''
    color_distance_map = [[0 for _ in range(class_length)] for _ in range(class_length)]

    for i in range(class_length):
        for j in range(i+1, class_length):
            color_distance_map[i][j] = distance_func(class_color[i], class_color[j])
            color_distance_map[j][i] = distance_func(class_color[i], class_color[j])
    
    return color_distance_map

if __name__ == "__main__":
    p1 = [205, 203, 202] # #CDCBCA ( 의자 등받이 부분 )
    p2 = [217, 217, 219] # #D9D9DB ( 의자 바닥 부분 )

    class_color = [[205, 203, 202], [90, 92, 105], [217, 217, 219], [172, 166, 153], [194, 181, 160], [92, 63, 43]]
    print(get_color_distance_map(class_color, len(class_color)))
    print(get_color_distance_map(class_color, len(class_color), distance_func=get_rgb_distance))
    
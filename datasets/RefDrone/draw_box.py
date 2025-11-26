import cv2
import numpy as np
import os
import pdb

def draw_boxes_from_annotations(image_name,dataset_name, annotations_list, num_lines_to_draw=1, box_color=(0, 255, 0), thickness=2,text_to_overlay=""):
    image_path = f"VisDrone2019-DET-{dataset_name}/images/{image_name}"
    if not os.path.exists(image_path):
        print(f"致命错误: 图片文件不存在于路径: {image_path}")
        exit()
    else:
        print(f"确认: 图片文件存在于路径: {image_path}")
    image = cv2.imread(image_path)

 
    for i, annotation_line in enumerate(annotations_list):
        if i >= num_lines_to_draw:
            break

        parts = annotation_line.split(',')
        if len(parts) < 4:
            print(f"警告: Annotation行 '{annotation_line}' 格式不正确，跳过。")
            continue
        
        
        try:
 
            x_min = int(float(parts[0]))
            y_min = int(float(parts[1]))
            width = int(float(parts[2]))
            height = int(float(parts[3]))

            
            
            x_max = x_min + width
            y_max = y_min + height

            # 获取图片实际尺寸
            img_height, img_width = image.shape[0], image.shape[1]

            # 检查边界框是否在图片范围内
            if not (0 <= x_min < img_width and 0 <= y_min < img_height and
                    x_max <= img_width and y_max <= img_height):
                print(f"DEBUG: 警告: 边界框 ({x_min},{y_min},{width},{height}) 超出图片尺寸 {img_width}x{img_height}，跳过绘制。")
                continue

            cv2.rectangle(image, (x_min, y_min), (x_max, y_max), box_color, thickness)
            print(f"DEBUG: 成功在图片上绘制了边界框: x_min={x_min}, y_min={y_min}, width={width}, height={height}")

        except ValueError:
            print(f"警告: Annotation行 '{annotation_line}' 包含非数字坐标，跳过。")
            continue


        # 在图片上添加文字
    if text_to_overlay: 
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_color =(255,255,255) 
        font_thickness = 2
        # 文字的起始坐标 (x,h-y)。这里设置为左下角，稍微向下偏移以免太贴边
        text_origin = (10,img_height- 30) 
        
        cv2.putText(image, text_to_overlay, text_origin, font, font_scale, font_color, font_thickness, cv2.LINE_AA)
    
    if image is not None:
        cv2.imshow("Image with Bounding Boxes", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        cv2.imwrite(f"./output/output-{image_name}", image)
        print(f"图片已保存到 ./output/output-{image_name}")
    else:
        print("未生成输出图片，请检查图片路径是否正确或文件已损坏。")     
          
    return image





def read_annotations_from_file(annotations_file_path):
    annotations = []
    try:
        with open(annotations_file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        annotations.append(line.strip())
    except FileNotFoundError:
        print(f"致命错误: 标注文件不存在于路径: {annotations_file_path}")
        exit()
    return annotations




if __name__ == "__main__":
    dataset = "test"
    file_name = "0000006_00611_d_0000002.jpg"
    b_num= 70
    
    name,_ =os.path.splitext(file_name)
    annotations_file_path = f"VisDrone2019-DET-{dataset}-dev/annotations/{name}.txt"
    annotations = read_annotations_from_file(annotations_file_path=annotations_file_path)

    draw_boxes_from_annotations(file_name, dataset, annotations, b_num)




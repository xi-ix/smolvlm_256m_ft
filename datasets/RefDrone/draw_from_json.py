import json
from draw_box import draw_boxes_from_annotations 

def read_json(file_path):
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None

def find_all_info(id ,json_data):
    for image in json_data["images"]:
        if image['id'] == id:
            file_name = image['file_name']
            caption = image['caption']
    for annotation in json_data["annotations"]:
        if annotation['image_id'] == id:
            bbox = annotation['bbox']
            # print(bbox)
    return file_name,caption,bbox

if __name__ == "__main__":
    dataset = "test"
    path_to_json_file = f"RefDrone_{dataset}_mdetr.json"
    
    id = 1
    data = read_json(path_to_json_file)
    file_name,caption,bbox = find_all_info(id,data)
    print(file_name,caption,bbox,sep='\n')
    
    bbox_str = [','.join(map(str, bbox))]  # 将bbox列表转换为逗号分隔的字符串
    
    draw_boxes_from_annotations(file_name, dataset, bbox_str,text_to_overlay=caption)
    
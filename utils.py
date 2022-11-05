import json

def load_json(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def process_annotation(annotations: list) -> list:
    # process annotations
    processed_annotations = []
    for i in range(len(annotations)):
        # get the annotation
        annotation = annotations[i]
        
        # Split the annotation into sentences
        annotation = annotation.split('.')
        
        # Strip leading and trailing spaces
        annotation = [x.strip() for x in annotation if x != '']
                
        processed_annotations += annotation        
    return processed_annotations


# Extract annotations from processed train and valid json files 
def generate_template_instructions(train_path: str, valid_path: str, output_path: str) -> None:
    train_json = load_json(train_path)
    valid_json = load_json(valid_path)
    
    # extract annotations from train and valid json files
    total_annotations = []
    
    for i in range(len(train_json)):
        total_annotations.append(train_json[i]['annotations'])
        
    for i in range(len(valid_json)):
        total_annotations.append(valid_json[i]['annotations'])
        
    # process annotations
    total_annotations = process_annotation(total_annotations)
    
    # make them into a unique element list
    total_annotations = list(set(total_annotations))
    
    # sort the list
    total_annotations.sort()
    
    # Write the annotations to a file
    with open(output_path, 'w') as f:
        for annotation in total_annotations:
            f.write(annotation + '\n')
            
            


if __name__ == '__main__':
    train_path = './simbot/train_processed.json'
    valid_path = './simbot/valid_processed.json'
    output_path = './template_instructions.txt'
    
    generate_template_instructions(train_path, valid_path, output_path)

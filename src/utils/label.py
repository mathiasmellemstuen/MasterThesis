from .label_type import Label_Type

class Label():
    def __init__(self, label : float, label_type : Label_Type):
        self.label = label
        self.label_type = label_type
    
    def __str__(self) -> str:
        return f"{self.label_type}: {self.label}"

def find_label(labels, label_type : Label_Type):
    for label in labels:
        if label.label_type == label_type:
            return label
        
    raise Exception("Could not find label")